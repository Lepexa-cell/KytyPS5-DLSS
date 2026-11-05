#version 450

// Scales a colour image into a colour target with a linear filter (BlitHelper::ScaleColor). Unlike
// vkCmdBlitImage it reads through a view, so the source is decoded in the format the guest wrote
// it with, not the format the host image was created with.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source; // linear, clamp to edge
layout(push_constant) uniform Scale {
	vec2 uv_scale; // source uv per target pixel
};

void main() {
	color = textureLod(source, gl_FragCoord.xy * uv_scale, 0.0);
}
