#ifndef KYTY_GRAPHICS_SHADER_BENCH_H_
#define KYTY_GRAPHICS_SHADER_BENCH_H_

namespace Libs::Graphics::ShaderBench {

// kyty_emulator --shader-bench <journal> [options]: replays a shader precompile journal through
// the recompiler offline and reports per-record translation cost, SPIR-V statistics and,
// optionally, driver compile time. argv[0] is "--shader-bench".
int Run(int argc, char* argv[]);

} // namespace Libs::Graphics::ShaderBench

#endif
