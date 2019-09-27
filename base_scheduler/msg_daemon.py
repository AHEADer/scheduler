class Daemon:
    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    # scheduler part begin
    def ask_grow(self):

        return

    def ask_shrink(self):
        return

    def recall_grow(self):
        return

    def recall_shrink(self):
        return

    def migrate(self):
        return

    def merge(self):
        return
    # scheduler part end

    # job part begin
    def receive_grow(self):
        return

    def receive_recall_grow(self):
        return

    def receive_shrink(self):
        return

    def receive_end(self):
        return

    # job part end
