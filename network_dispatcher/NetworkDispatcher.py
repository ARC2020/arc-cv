import threading

from modules.arc_comms import NetworkPackage

class NetworkDispatcher(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.dispatch_pipe = None
        self.stop_condition = False

    def attach_dispatch_pipe(self, dispatch_pipe):
        self.dispatch_pipe = dispatch_pipe
    
    def stop():
        self.stop_condition = True

    def run(self):
        while True:
                if self.stop_condition:
                    break
                if self.dispatch_pipe == None:
                    continue
                if not self.dispatch_pipe.empty():
                    networkPackage = self.dispatch_pipe.get_nowait()
                    # print(networkPackage)