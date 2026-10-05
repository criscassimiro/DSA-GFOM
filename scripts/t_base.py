import time, json
from comum import *
t=time.time(); o=walk_forward("RF",PRELIMINAR["RF"],"2022-01-01","2025-12-01"); print("tempo",time.time()-t)
print(json.dumps(avaliar(o),indent=1,default=float))
