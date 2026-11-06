"""Download and verify the published inputs for the defined-community benchmark."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile
import io
import re
import json
import time
import struct
from zlib import crc32

ROOT = Path(__file__).resolve().parent
ARCHIVE = "AGORA2/version2.01/sbml_files_fixed/zipped/AGORA2_models/AGORA2_SBML.zip"
URL = "https://vmh.life/api/reconstructionFiles?download=" + quote(ARCHIVE, safe="")
SIZE = 2170499647



class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, SIZE)[whence] + offset
        if self.position < 0:
            raise ValueError("Invalid seek")
        return self.position

    def read(self, size=-1):
        remaining = max(0, SIZE - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        start, end = self.position, self.position + size - 1
        expected = f"bytes {start}-{end}/{SIZE}"
        for attempt in range(3):
            try:
                request = Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "BioRxnGen benchmark"})
                with urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise OSError(f"Unexpected byte range {response.status}, {response.headers.get('Content-Range')}")
                    result = response.read()
                if len(result) != size:
                    raise OSError(f"Expected {size} bytes, received {len(result)}")
                self.position += size
                return result
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)



def raw_model(archive, entry):
    if entry.compress_type != 9:
        return archive.read(entry)
    from inflate64 import Inflater

    remote = RemoteZip()
    remote.seek(entry.header_offset)
    header = remote.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP member")
    filename, extra = struct.unpack_from("<HH", header, 26)
    remote.seek(filename + extra, 1)
    data = Inflater().inflate(remote.read(entry.compress_size))
    if len(data) != entry.file_size or crc32(data) != entry.CRC:
        raise ValueError(f"Archive CRC mismatch for {entry.filename}")
    return data


def utf8_model(data, record):
    if not record.get("original_sha256"):
        return data
    text = data.decode("utf-8", errors="surrogateescape")
    return re.sub(r'(?<=\bname=")[^"]+(?=")',
                  lambda m: re.sub(r"[\udc80-\udcff]", lambda c: chr(ord(c.group()) - 0xdc00), m.group()),
                  text).encode("utf-8")


def verify_models(out):
    for record in json.loads((ROOT / "data/models.json").read_text()):
        path = out / record["file"]
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Unexpected AGORA2 SBML file {path}")


def fetch_models(out):
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "data/models.json").read_text())
    pending = [r for r in records if not (out / r["file"]).exists()]
    if pending:
        with ZipFile(RemoteZip()) as archive:
            entries = {r["file"]: archive.getinfo(r["archive_entry"]) for r in pending}
        def fetch(record):
            entry = entries[record["file"]]
            if entry.compress_type == 9:
                data = raw_model(None, entry)
            else:
                with ZipFile(RemoteZip()) as archive:
                    data = raw_model(archive, entry)
            if sha256(data).hexdigest() != record.get("original_sha256", record["sha256"]):
                raise ValueError(f"Source SBML checksum mismatch for {record['file']}")
            model = utf8_model(data, record)
            if sha256(model).hexdigest() != record["sha256"]:
                raise ValueError(f"SBML encoding mismatch for {record['file']}")
            (out / record["file"]).write_bytes(model)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch, pending))
    verify_models(out)
    print(f"Verified {len(records)} AGORA2 models.")
