import tensorflow as tf
from AIConsumer import AIConsumer

def main(_):
    frameConsumer = AIConsumer()
    frameConsumer.begin()


if __name__ == "__main__":
    tf.app.run()