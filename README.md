# fishtools

[![Install and run](https://github.com/chaichontat/fishtools/actions/workflows/test.yml/badge.svg)](https://github.com/chaichontat/fishtools/actions/workflows/test.yml)

Tools for FISH analysis

## Installation

```sh
pip install -e .
```

## Compression

- `fishtools compress` is a command-line interface (CLI) tool that converts TIFF, JP2, and DAX image files to JPEG XL (JXL) files.
- `fishtools decompress` is a CLI tool that converts JXL files back to DAX files.

![Comparison of different compression quality](https://github.com/chaichontat/fishtools/assets/34997334/95230a08-4817-433d-a98d-67b5c442439d)

#### Usage

To use `fishtools`, simply run the `fishtools` command followed by the subcommand and path to the directory containing TIFF, JP2, or DAX files that you want to convert:

```sh
fishtools compress path/to/directory
```

By default, `fishtools compress` will convert all TIFF, JP2, and DAX files in the specified directory and its subdirectories. The converted JXL files will be saved in the same directory as the original files with the same name but with a `.jxl` extension.

You can also specify the quality level of the JXL files using the `--quality` or `-q` option. The quality level should be an integer between -inf and 100, where 100 is lossless. The default quality level is 99 (about 10x reduction in file size).

When the lossless option is selected, the output file is a `.tif` file with JPEG-XR encoding so that the file can be opened in ImageJ/BioFormats.

> BioFormats in ImageJ does not support JPEG-XL yet.
> It does support JPEG-XR which provides the same performance for lossless compression.
> JPEG-XR does not compress >8-channel images (compression scheme not in the specification).

If you want to delete the original files after conversion, you can use the `--delete` or `-d` option.

To use `fishtools decompress`, simply run the `fishtools decompress` command followed by the path to the JXL file or directory containing JXL files that you want to convert.
By default, `fishtools decompress` will convert all JXL files in the specified directory and its subdirectories.
The converted DAX files will be saved in the same directory as the original JXL files with the same name but with a `.dax` extension.

#### Examples

Convert all TIFF, JP2, and DAX files in a directory and its subdirectories to lossless TIFFs with JPEG-XR encoding:

```sh
fishtools compress path/to/directory
```

Convert all TIFF, JP2, and DAX files in a directory to JPEG-XL files and delete the original files:

```sh
fishtools compress path/to/directory --quality 99 --delete
```

Convert a single JXL file to DAX:

```sh
fishtools decompress path/to/file.jxl
```

Convert all JXL files in a directory to DAX:

```sh
fishtools decompress path/to/directory
```

## Probe ordering checklist

1. Verify simulation
2. BLAST some probes, make sure orientation is Plus/Minus.
3. Delete all old final files, both remote and local.
4. Run the script one last time.
5. Download said file and open to copy/paste into the Excel order sheet.
6. Save as a new file with today's date.
7. In the email, upload and redownload, verify that it's the same file.

## License

This project is licensed under the [MIT License](LICENSE).
