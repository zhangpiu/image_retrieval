
import struct
from feature_pb2 import ImageFeatureDump


feature = ImageFeatureDump()
feature.value.extend([1,2,3])
feature.label = 1
data = feature.SerializeToString()
size = struct.pack('<Q', len(data))

with open('2.txt', 'wb') as file:
  file.write(size)
  file.write(data)
