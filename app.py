from AIConsumer import AIConsumer

def main():
    frameConsumer = AIConsumer("fcn-resnet18-deepscene")
    frameConsumer.start()

if __name__ == "__main__":
    main()
