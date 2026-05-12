import argparse
import csv
import re
import sys
import zipfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape, quoteattr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "outputs" / "result.csv"
MAX_EXCEL_ROWS = 1_048_576
MAX_EXCEL_COLS = 16_384
MAX_CELL_CHARS = 32_767
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
INVALID_SHEET_CHARS = re.compile(r"[:\\/?*\[\]]")


def col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def clean_text(value: str) -> str:
    value = value[:MAX_CELL_CHARS]
    return "".join(
        char
        for char in value
        if char in "\t\n\r" or ord(char) >= 32
    )


def xml_text(value: str) -> str:
    value = clean_text(value)
    attrs = ' xml:space="preserve"' if value[:1].isspace() or value[-1:].isspace() else ""
    return f"<t{attrs}>{escape(value)}</t>"


def looks_like_number(value: str) -> bool:
    if not value or value != value.strip():
        return False
    if not NUMBER_RE.fullmatch(value):
        return False

    mantissa = value.lstrip("+-").split("e", 1)[0].split("E", 1)[0]
    integer_part = mantissa.split(".", 1)[0]
    if len(integer_part) > 1 and integer_part.startswith("0"):
        return False

    try:
        Decimal(value)
    except InvalidOperation:
        return False
    return True


def clean_sheet_name(name: str) -> str:
    name = INVALID_SHEET_CHARS.sub("_", name).strip("'").strip()
    return (name or "Sheet1")[:31]


def read_csv(path: Path, encoding: Optional[str]) -> Tuple[List[List[str]], str]:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "gb18030"]
    last_error: Optional[UnicodeDecodeError] = None

    for candidate in encodings:
        try:
            with path.open("r", encoding=candidate, newline="") as file:
                rows = [row for row in csv.reader(file)]
            return rows, candidate
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise RuntimeError("No encoding candidates were provided.")


def validate_size(rows: Sequence[Sequence[str]]) -> int:
    if len(rows) > MAX_EXCEL_ROWS:
        raise ValueError(f"CSV has {len(rows)} rows; Excel supports at most {MAX_EXCEL_ROWS}.")

    max_cols = max((len(row) for row in rows), default=0)
    if max_cols > MAX_EXCEL_COLS:
        raise ValueError(f"CSV has {max_cols} columns; Excel supports at most {MAX_EXCEL_COLS}.")
    return max_cols


def column_widths(rows: Sequence[Sequence[str]], max_cols: int) -> List[int]:
    widths = [8] * max_cols
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], min(len(clean_text(value)) + 2, 60))
    return widths


def iter_cells(
    row: Sequence[str],
    row_idx: int,
    infer_numbers: bool,
) -> Iterable[str]:
    for col_idx, value in enumerate(row, start=1):
        if value == "":
            continue

        ref = f"{col_name(col_idx)}{row_idx}"
        if row_idx > 1 and infer_numbers and looks_like_number(value):
            yield f'<c r="{ref}"><v>{escape(value)}</v></c>'
        else:
            style = ' s="1"' if row_idx == 1 else ""
            yield f'<c r="{ref}" t="inlineStr"{style}><is>{xml_text(value)}</is></c>'


def worksheet_xml(rows: Sequence[Sequence[str]], sheet_name: str, infer_numbers: bool) -> str:
    max_cols = validate_size(rows)
    last_col = col_name(max(max_cols, 1))
    last_row = max(len(rows), 1)
    dimension = f"A1:{last_col}{last_row}"

    widths = column_widths(rows, max_cols)
    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(widths, start=1)
    )
    cols_xml = f"<cols>{cols_xml}</cols>" if cols_xml else ""

    freeze_xml = (
        '<sheetViews><sheetView tabSelected="1" workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        if rows
        else ""
    )
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(iter_cells(row, row_idx, infer_numbers))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')

    auto_filter = f'<autoFilter ref="{dimension}"/>' if rows and max_cols else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        f"{freeze_xml}"
        f"{cols_xml}"
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        f"{auto_filter}"
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>'
        "</worksheet>"
    )


def workbook_xml(sheet_name: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        f'<sheet name={quoteattr(sheet_name)} sheetId="1" r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>'
        "</fonts>"
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        "</cellXfs>"
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '<dxfs count="0"/>'
        '<tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>'
        "</styleSheet>"
    )


def content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )


def package_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def workbook_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )


def app_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Python</Application>"
        "</Properties>"
    )


def core_xml() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>export_result_to_excel.py</dc:creator>"
        "<cp:lastModifiedBy>export_result_to_excel.py</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def write_xlsx(rows: Sequence[Sequence[str]], output_path: Path, sheet_name: str, infer_numbers: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_name = clean_sheet_name(sheet_name)

    files = {
        "[Content_Types].xml": content_types_xml(),
        "_rels/.rels": package_rels_xml(),
        "docProps/app.xml": app_xml(),
        "docProps/core.xml": core_xml(),
        "xl/workbook.xml": workbook_xml(sheet_name),
        "xl/_rels/workbook.xml.rels": workbook_rels_xml(),
        "xl/styles.xml": styles_xml(),
        "xl/worksheets/sheet1.xml": worksheet_xml(rows, sheet_name, infer_numbers),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        for name, content in files.items():
            workbook.writestr(name, content)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CSV file to an Excel .xlsx workbook.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help="CSV file to export. Defaults to outputs/result.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .xlsx path. Defaults to the CSV path with .xlsx extension.",
    )
    parser.add_argument("--sheet-name", default=None, help="Worksheet name. Defaults to the CSV file stem.")
    parser.add_argument("--encoding", default=None, help="CSV encoding. Defaults to utf-8-sig/utf-8/gb18030 auto try.")
    parser.add_argument(
        "--no-infer-numbers",
        action="store_true",
        help="Write all cells as text instead of converting numeric-looking values to numbers.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    csv_path = csv_path.resolve()

    output_path = Path(args.output) if args.output else csv_path.with_suffix(".xlsx")
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()

    rows, encoding = read_csv(csv_path, args.encoding)
    max_cols = validate_size(rows)
    sheet_name = args.sheet_name or csv_path.stem
    write_xlsx(rows, output_path, sheet_name, infer_numbers=not args.no_infer_numbers)

    print(f"Wrote {output_path}")
    print(f"Rows: {len(rows)}, columns: {max_cols}, source encoding: {encoding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
