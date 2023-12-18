# darkblur
A simple utility to detect dark or blurry micrographs processed using Warp

```commandline
git clone https://github.com/shahpnmlab/darkblur.git
cd darkblur
conda create --name darkblur python=3.10 -y
conda activate darkblur
pip install . 
```
# Running
```commandline
darkblur -mrc /path/to/mrc -xml /path/to/warp_xmls
```