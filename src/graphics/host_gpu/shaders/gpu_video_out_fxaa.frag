#version 450

// Presents one mip level of a prepared frame with FXAA (see PresentFilter): finds the local edge
// direction from luma, walks along the edge to both ends, and resamples across it, plus the
// subpixel blend of FXAA 3.11's quality preset. Steps are one target pixel in source texels, so
// a frame up to twice the window size is smoothed and resampled in the same pass.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source; // linear, clamp to edge
layout(push_constant) uniform Fxaa {
	ivec2 size;  // extent of the source level
	vec2  scale; // source texels per target pixel
	int   level;
	ivec2 offset; // target pixel of the frame's top-left corner (letterbox)
};

const float EdgeThresholdMin = 0.0312;
const float EdgeThreshold    = 0.125;
const float SubpixelQuality  = 0.75;
const int   Iterations       = 12;
const float Quality[Iterations] =
    float[](1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);

vec4 Tap(vec2 uv) {
	return textureLod(source, uv, float(level));
}

// Perceptual luma whether the frame is stored linear or gamma-encoded closely enough for edges.
float Luma(vec3 rgb) {
	return sqrt(max(dot(rgb, vec3(0.299, 0.587, 0.114)), 0.0));
}

float LumaAt(vec2 uv) {
	return Luma(Tap(uv).rgb);
}

void main() {
	const vec2 texel = 1.0 / vec2(size);
	const vec2 inv   = texel * max(scale, vec2(1.0));
	const vec2 uv    = (gl_FragCoord.xy - vec2(offset)) * scale * texel;

	const vec4  center = Tap(uv);
	const float lM     = Luma(center.rgb);
	const float lD     = LumaAt(uv + vec2(0.0, -inv.y));
	const float lU     = LumaAt(uv + vec2(0.0, inv.y));
	const float lL     = LumaAt(uv + vec2(-inv.x, 0.0));
	const float lR     = LumaAt(uv + vec2(inv.x, 0.0));
	const float lMin   = min(lM, min(min(lD, lU), min(lL, lR)));
	const float lMax   = max(lM, max(max(lD, lU), max(lL, lR)));
	const float range  = lMax - lMin;
	if (range < max(EdgeThresholdMin, lMax * EdgeThreshold)) {
		color = center;
		return;
	}

	const float lDL = LumaAt(uv + vec2(-inv.x, -inv.y));
	const float lUR = LumaAt(uv + vec2(inv.x, inv.y));
	const float lUL = LumaAt(uv + vec2(-inv.x, inv.y));
	const float lDR = LumaAt(uv + vec2(inv.x, -inv.y));

	const float lDownUp       = lD + lU;
	const float lLeftRight    = lL + lR;
	const float lLeftCorners  = lDL + lUL;
	const float lDownCorners  = lDL + lDR;
	const float lRightCorners = lDR + lUR;
	const float lUpCorners    = lUR + lUL;

	const float edgeHorizontal = abs(-2.0 * lL + lLeftCorners) +
	                             abs(-2.0 * lM + lDownUp) * 2.0 + abs(-2.0 * lR + lRightCorners);
	const float edgeVertical = abs(-2.0 * lU + lUpCorners) + abs(-2.0 * lM + lLeftRight) * 2.0 +
	                           abs(-2.0 * lD + lDownCorners);
	const bool horizontal = edgeHorizontal >= edgeVertical;

	const float luma1     = horizontal ? lD : lL;
	const float luma2     = horizontal ? lU : lR;
	const float gradient1 = luma1 - lM;
	const float gradient2 = luma2 - lM;
	const bool  steepest1 = abs(gradient1) >= abs(gradient2);
	const float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

	float stepLength = horizontal ? inv.y : inv.x;
	float lumaLocalAverage;
	if (steepest1) {
		stepLength       = -stepLength;
		lumaLocalAverage = 0.5 * (luma1 + lM);
	} else {
		lumaLocalAverage = 0.5 * (luma2 + lM);
	}

	vec2 current = uv;
	if (horizontal) {
		current.y += stepLength * 0.5;
	} else {
		current.x += stepLength * 0.5;
	}
	const vec2 offset = horizontal ? vec2(inv.x, 0.0) : vec2(0.0, inv.y);
	vec2       uv1    = current - offset;
	vec2       uv2    = current + offset;
	float      end1   = LumaAt(uv1) - lumaLocalAverage;
	float      end2   = LumaAt(uv2) - lumaLocalAverage;
	bool       reached1 = abs(end1) >= gradientScaled;
	bool       reached2 = abs(end2) >= gradientScaled;
	if (!reached1) {
		uv1 -= offset;
	}
	if (!reached2) {
		uv2 += offset;
	}
	for (int i = 2; i < Iterations && !(reached1 && reached2); i++) {
		if (!reached1) {
			end1 = LumaAt(uv1) - lumaLocalAverage;
		}
		if (!reached2) {
			end2 = LumaAt(uv2) - lumaLocalAverage;
		}
		reached1 = abs(end1) >= gradientScaled;
		reached2 = abs(end2) >= gradientScaled;
		if (!reached1) {
			uv1 -= offset * Quality[i];
		}
		if (!reached2) {
			uv2 += offset * Quality[i];
		}
	}

	const float distance1  = horizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
	const float distance2  = horizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);
	const bool  direction1 = distance1 < distance2;
	const float thickness  = distance1 + distance2;
	const float pixelOffset = -min(distance1, distance2) / thickness + 0.5;
	const bool  centerSmaller   = lM < lumaLocalAverage;
	const bool  correctVariation = ((direction1 ? end1 : end2) < 0.0) != centerSmaller;
	float       finalOffset      = correctVariation ? pixelOffset : 0.0;

	const float lumaAverage =
	    (1.0 / 12.0) * (2.0 * (lDownUp + lLeftRight) + lLeftCorners + lRightCorners);
	const float subpixel1 = clamp(abs(lumaAverage - lM) / range, 0.0, 1.0);
	const float subpixel2 = (-2.0 * subpixel1 + 3.0) * subpixel1 * subpixel1;
	finalOffset           = max(finalOffset, subpixel2 * subpixel2 * SubpixelQuality);

	vec2 final_uv = uv;
	if (horizontal) {
		final_uv.y += finalOffset * stepLength;
	} else {
		final_uv.x += finalOffset * stepLength;
	}
	color = Tap(final_uv);
}
