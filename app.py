from pipeline import SegProducer, FrameProducer
from queue import Queue
from network_dispatcher import NetworkDispatcher

def main():
    width = 1280
    height = 720

    pipeline = Queue(5)
    dispatch_pipeline = Queue(5)

    frameConsumer = SegProducer("fcn-resnet18-mhp", width, height)
    frameProducer = FrameProducer(width, height)
    networkDispatcher = NetworkDispatcher

    frameProducer.attach_pipe(pipeline)
    frameConsumer.attach_pipe(pipeline)
    frameConsumer.attach_dispatch_pipe(dispatch_pipeline)
    networkDispatcher.attach_dispatch_pipe(dispatch_pipeline)

    frameProducer.start()
    frameConsumer.start()
    networkDispatcher.start()



    wait = input()

    print("CV-APP: Stopping...")
    frameConsumer.stop()
    frameConsumer.join()
    frameProducer.stop()
    frameProducer.join()
    print("CV-APP: All Threads Halted. Exiting...")
    

if __name__ == "__main__":
    main()
