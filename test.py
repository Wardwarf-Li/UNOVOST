import json
import numpy as np
from numpy import array as arr
from pycocotools.mask import decode, encode
from PIL import Image
with open('00000.json', 'r') as f:
    proposals = json.load(f)

# curr_prop = dict()
# curr_prop['seg'] = [prop["segmentation"] for prop in proposals]
#
# print(curr_prop)
# print(len(proposals))


curr_prop = dict()
curr_prop['seg'] = [prop["segmentation"] for prop in proposals]
curr_prop['fwd'] = [prop["forward_segmentation"] if "forward_segmentation" in prop.keys() else None for prop in
                    proposals]

curr_prop['reid'] = [arr(prop["ReID"]) if "ReID" in prop.keys() else np.inf * np.ones((128)) for prop in
                         proposals]
curr_prop['score'] = arr([prop["score"] for prop in proposals])
curr_prop['id'] = np.arange(0, len(proposals))
curr_prop['mask'] = [decode(seg) for seg in curr_prop['seg']]


print(curr_prop['seg'])

rs = encode(curr_prop['mask'][0])
print(rs)
ms = decode(rs)
print(ms)
# for i in range(len(curr_prop['seg'])):
#
#     mask = Image.fromarray(curr_prop['mask'][i]*255)
#     mask.show()


