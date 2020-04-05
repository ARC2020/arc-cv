from pipeline import SegProducer, FrameProducer
from queue import Queue

def main():
    width = 1280
    height = 720

    pipeline = Queue(5)
    frameConsumer = SegProducer("fcn-resnet18-mhp", width, height)
    frameProducer = FrameProducer(width, height)

    frameProducer.attach_pipe(pipeline)
    frameProducer.start()

    frameConsumer.attach_pipe(pipeline)
    frameConsumer.start()

    wait = input()

    print("CV-APP: Stopping...")
    frameConsumer.stop()
    frameConsumer.join()
    frameProducer.stop()
    frameProducer.join()
    print("CV-APP: All Threads Halted. Exiting...")
    

if __name__ == "__main__":
    main()
