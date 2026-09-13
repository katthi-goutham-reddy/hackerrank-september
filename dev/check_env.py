import os
for k in os.environ:
    if 'KEY' in k or 'API' in k or 'TOKEN' in k or 'SECRET' in k:
        print(f"{k}: len={len(os.environ[k])}, prefix={os.environ[k][:5]}...")
