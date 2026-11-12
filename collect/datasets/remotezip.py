import io, sys, urllib.request, zipfile
URL = "https://zenodo.org/api/records/15083865/files/ZENODO_Penelope_vs2.zip/content"
class RangeFile(io.RawIOBase):
    def __init__(s, url):
        s.url = urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).geturl()
        s.size = int(urllib.request.urlopen(urllib.request.Request(s.url, method="HEAD")).headers["Content-Length"]); s.pos = 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, whence=0):
        s.pos = off if whence == 0 else (s.pos + off if whence == 1 else s.size + off); return s.pos
    def readinto(s, buf):
        n = len(buf)
        if n == 0 or s.pos >= s.size: return 0
        req = urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{min(s.pos+n, s.size)-1}"})
        b = urllib.request.urlopen(req).read(); buf[:len(b)] = b; s.pos += len(b); return len(b)
def open_zip(): return zipfile.ZipFile(io.BufferedReader(RangeFile(URL), buffer_size=4 << 20))
if __name__ == "__main__":
    z = open_zip(); out = sys.argv[1]
    for pat in sys.argv[2:]:
        for n in z.namelist():
            if pat in n and not n.endswith("/"):
                z.extract(n, out); print("got", n)
