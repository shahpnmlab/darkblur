from pathlib import Path
import typer
from darkblur.utils import is_image_dark, is_image_blurry, is_image_obstructed

darkblur = typer.Typer()


@darkblur.command(no_args_is_help=True)
def analyze_images(
        path_to_mrc: str = typer.Option(..., "--mrc", "-m", help="Path to the directory containing MRC files.")):
    mrcfiles = list(Path(path_to_mrc).glob("*.mrc"))  # Convert generator to list for multiple iterations
    log_file_path = Path(path_to_mrc) / "log.txt"

    with log_file_path.open("w") as log_file:
        for f in mrcfiles:
            if is_image_dark(f):
                log_file.write(f'{f} - DARK\n')
                continue  # Skip further checks for this file

            if is_image_obstructed(f):
                log_file.write(f'{f} - OBSTRUCTION in FOV\n')
                continue  # Skip blur check for this file

            if is_image_blurry(f):
                log_file.write(f'{f} - BLURRED\n')

    typer.echo(f"Analysis completed. Log file created at {log_file_path}")


if __name__ == "__main__":
    darkblur()