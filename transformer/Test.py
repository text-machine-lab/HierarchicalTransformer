print('Hello world')
import sys

print(sys.path)
print(__name__)
print(__name__.split('.')[0])

top_package = __import__(__name__.split('.')[0])