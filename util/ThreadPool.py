import sys
IS_PY2 = sys.version_info < (3, 0)

if IS_PY2:
    from Queue import Queue
else:
    from queue import Queue

import threading
from threading import Thread


class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks, session):
        Thread.__init__(self)
        self.tasks = tasks
        self.session = session
        self.daemon = True
        self._stop_event = threading.Event()
        self.start()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, self.session)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """
    def __init__(self, num_threads, sessions):
        try:
            if len(sessions) != num_threads:
                raise ValueError("Not enough sessions for threads")
            self.tasks = Queue(num_threads)
            self.workers = []
            for index in range(num_threads):
                self.workers.append(Worker(self.tasks, sessions[index]))
        except ValueError as e:
            print(e)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def shutdown(self):
        for index in range(num_threads):
            self.workers[index].stop()
            self.workers[index].join()

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()