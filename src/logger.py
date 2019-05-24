class Logger:
    def __init__(self, **kwargs):
        self.log_to_stdout = kwargs.get("log_to_stdout", True)
        self.log_to_tb = kwargs.get("log_to_tb", True)
        self.tb_writer = kwargs.get("tb_writer", True)

    def log(self, msg, **kwargs):
        label = kwargs.get("label", "output")
        epoch = kwargs.get("epoch", 0)

        if self.log_to_stdout:
            print(msg)

        if self.log_to_tb:
            self.tb_writer.add_text(label, msg, epoch)
