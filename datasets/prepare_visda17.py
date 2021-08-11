import urllib.request
import tarfile
import pathlib


def main():
    root = pathlib.Path(__file__).parent / "visda17"
    root.mkdir(exist_ok=True)

    files = [
        "train.tar",
        "validation.tar",
        "test.tar",
    ]

    for f in files:
        print(f"[*] Downloading {f}...")
        archive, _ = urllib.request.urlretrieve(
            f"http://csr.bu.edu/ftp/visda17/clf/{f}", (root / f).as_posix()
        )
        print(f"[*] Extracting {f}...")
        tarfile.extract(archive, root)


if __name__ == "__main__":
    main()
