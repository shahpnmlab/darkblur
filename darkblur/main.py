from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from darkblur.utils import is_image_dark, is_image_blurry

darkblur = typer.Typer()


@darkblur.command(no_args_is_help=True)
def analyze_images(
        path_to_mrc: str = typer.Option(..., "--mrc", "-mrc", help="Path to the directory containing MRC files."),
        path_to_xml: Optional[str] = typer.Option(None, "--xml", "-xml",
                                                  help="Path to the directory containing XML files.")
):
    mrcfiles = Path(path_to_mrc).glob("*.mrc")
    xml_dir = Path(path_to_xml) if path_to_xml else None
    xbn = {f.stem for f in xml_dir.glob("*.xml")} if xml_dir else set()
    log_entries = []

    with tqdm(mrcfiles, desc="Analyzing Images", unit=" file(s)") as pbar:
        for mfile in pbar:
            bn = mfile.stem
            xml_file = xml_dir / f'{bn}.xml' if bn in xbn and xml_dir else None
            dark = is_image_dark(mfile, xml_file)
            blurry = False
            if not dark:  # Skip blurry check if image is dark
                blurry = is_image_blurry(mfile, xml_file)
            status = "dark" if dark else ("blurry" if blurry else "none")
            log_entries.append(f"{mfile}\t{status}")
            pbar.set_postfix(file=bn)

        # Writing the log file
    log_file_path = Path(path_to_xml) / "DarkBlurAnalysis_log.txt"
    with log_file_path.open("w") as log_file:
        for entry in log_entries:
            log_file.write(entry + "\n")

    typer.echo(f"Analysis completed. Log file created at {log_file_path}")


if __name__ == "__main__":
    darkblur()
