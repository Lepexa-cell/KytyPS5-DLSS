#ifndef COMMON_CONSOLECOLORS_H_
#define COMMON_CONSOLECOLORS_H_

namespace Common::ConsoleColors {

// Colors the emulator's console output by line category (errors, warnings, audio, shaders,
// memory). Only when stdout is a console window and kyty_console.exe sits next to the emulator:
// stdout/stderr then go through a pipe to that helper, which writes the colored lines to the same
// console. The helper is its own process, so it still prints the last lines when the emulator
// dies. A redirected stdout (pipe, file) stays as it is. KYTY_CONSOLE_COLORS=0 turns it off.
void Install();

} // namespace Common::ConsoleColors

#endif /* COMMON_CONSOLECOLORS_H_ */
