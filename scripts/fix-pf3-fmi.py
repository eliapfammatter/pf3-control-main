import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <fmu_file>")
        sys.exit(1)

    fmu_path = Path(sys.argv[1])
    missing_files_dir = Path("data/pf3/missing_files")

    if not fmu_path.exists():
        print(f"Error: FMU not found: {fmu_path}")
        sys.exit(1)

    if not missing_files_dir.exists():
        print(f"Error: directory not found: {missing_files_dir}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract
        with zipfile.ZipFile(fmu_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)

        # Copy all files into resources/Simsen/Model/DAT/
        dat_dir = temp_path / "resources" / "Simsen" / "Model" / "DAT"
        if not dat_dir.exists():
            print(f"Error: FMU structure missing {dat_dir}")
            sys.exit(1)

        for missing_file in missing_files_dir.iterdir():
            if missing_file.is_file():
                shutil.copy(missing_file, dat_dir / missing_file.name)
                print(f"Copied {missing_file.name}")

        # Repack
        temp_output = fmu_path.with_suffix(".fmu.tmp")
        with zipfile.ZipFile(temp_output, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    zip_ref.write(file_path, file_path.relative_to(temp_path))

        shutil.move(str(temp_output), str(fmu_path))
        print(f"Fixed: {fmu_path}")


if __name__ == "__main__":
    main()
