# fishtools

Tools for FISH analysis

## Installation

```sh
pip install -e .
```

## Compression

### `tojxl` and `todax`

- `tojxl` is a command-line interface (CLI) tool that converts TIFF, JP2, and DAX image files to JPEG XL (JXL) files.
- `todax` is a CLI tool that converts JXL files back to DAX files.

#### Usage

To use `tojxl`, simply run the `tojxl` command followed by the path to the directory containing TIFF, JP2, or DAX files that you want to convert:

```sh
tojxl path/to/directory
```

By default, `tojxl` will convert all TIFF, JP2, and DAX files in the specified directory and its subdirectories. The converted JXL files will be saved in the same directory as the original files with the same name but with a `.jxl` extension.

You can also specify the quality level of the JXL files using the `--quality` or `-q` option. The quality level should be an integer between -inf and 100, where 100 is lossless. The default quality level is 99.

If you want to delete the original files after conversion, you can use the `--delete` or `-d` option.

To use `todax`, simply run the `todax` command followed by the path to the JXL file or directory containing JXL files that you want to convert.
By default, `todax` will convert all JXL files in the specified directory and its subdirectories.
The converted DAX files will be saved in the same directory as the original JXL files with the same name but with a `.dax` extension.

#### Examples

Convert all TIFF, JP2, and DAX files in a directory:

```sh
tojxl path/to/directory
```

Convert all TIFF, JP2, and DAX files in a directory with a quality level of 80 and delete the original files:

```sh
tojxl path/to/directory --quality 80 --delete
```

Convert a single JXL file to DAX:

```sh
todax path/to/file.jxl
```

Convert all JXL files in a directory to DAX:

```sh
todax path/to/directory
```

## License

This project is licensed under the [MIT License](LICENSE).
