from pipeline import SegProducer
from pipeline import FrameProducer
from network_dispatcher import NetworkDispatcher
from queue import Queue

def main():
    width = 1280
    height = 720

    pipeline_queue = Queue(5)
    dispatch_pipeline_queue = Queue(5)

    frameConsumer = SegProducer("fcn-resnet18-mhp", width, height)
    frameProducer = FrameProducer(width, height)
    networkDispatcher = NetworkDispatcher()

    frameProducer.attach_pipe(pipeline_queue)
    frameConsumer.attach_pipe(pipeline_queue)
    frameConsumer.attach_dispatch_pipe(dispatch_pipeline_queue)
    networkDispatcher.attach_dispatch_pipe(dispatch_pipeline_queue)

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
