#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_

namespace Libs::Graphics {

// Debugging aid (Windows): with KYTY_DEBUG_SAMPLE_GPU=<period in microseconds> (values below 100
// select 500), a helper thread samples the calling thread's call stack and writes each 10-second
// window's stacks to sample-<name>-<window>.txt in the working directory, as module+offset
// frames (innermost first) for llvm-symbolizer. The sampled thread stops only while its
// registers and the top of its stack are copied. Call it from the thread to sample.
void StartThreadSampler(const char* name);

// KYTY_GPU_CORE=<physical core>: gives the calling thread that core to itself (Windows). Other
// threads pass their own variable (KYTY_SPEC_CORE, KYTY_SUBMIT_CORE) and log label.
// `value`: the switch's text (KYTY_GETENV("KYTY_GPU_CORE") etc.; the share build needs the name there).
void ReserveCoreForCurrentThread(const char* value, const char* label);

// Debugging aid (Windows): with KYTY_DEBUG_DUMP_THREADS=<seconds>, a helper thread writes the call
// stacks of all the process's threads to thread-dump-<n>.txt every <seconds>, one "# tid=... name=..."
// line and one sampler-format stack per thread, to see what a stalled game waits on.
void StartThreadDumper();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
