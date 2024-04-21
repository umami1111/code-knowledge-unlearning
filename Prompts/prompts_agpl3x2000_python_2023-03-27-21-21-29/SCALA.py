# Must emulate terminal, otherwise `scalac` hangs on a call to `stty`
class Executor(JavaExecutor):
