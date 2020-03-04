from AIConsumer import AIConsumer

def main(_):
    frameConsumer = AIConsumer("fcn-resnet18-deepscene")
    frameConsumer.start()

if __name__ == "__main__":
    main()
