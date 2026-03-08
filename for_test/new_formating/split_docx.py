"""Split a .docx file into parts of N pages each using Word COM automation."""
import os
import sys
import time
import win32com.client
from win32com.client import constants

INPUT_FILE = r"C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\original_new.docx"
OUTPUT_DIR = r"C:\Users\Urdul\Desktop\project\translator\for_test\new_formating\section"
PAGES_PER_PART = 50

def split_docx():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    word = win32com.client.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0  # wdAlertsNone

    try:
        doc = word.Documents.Open(INPUT_FILE, ReadOnly=True)
        time.sleep(2)  # Let Word finish pagination

        # Force repagination
        doc.Repaginate()
        total_pages = doc.ComputeStatistics(2)  # wdStatisticPages = 2
        print(f"Total pages: {total_pages}")

        if total_pages == 0:
            print("Error: Could not determine page count")
            doc.Close(0)
            return

        num_parts = (total_pages + PAGES_PER_PART - 1) // PAGES_PER_PART
        print(f"Will create {num_parts} parts of up to {PAGES_PER_PART} pages each")

        for part_idx in range(num_parts):
            start_page = part_idx * PAGES_PER_PART + 1
            end_page = min((part_idx + 1) * PAGES_PER_PART, total_pages)
            part_num = part_idx + 1
            output_path = os.path.join(OUTPUT_DIR, f"original_new_part{part_num}.docx")

            print(f"Part {part_num}: pages {start_page}-{end_page} -> {output_path}")

            # Re-open document fresh for each part to avoid selection issues
            src = word.Documents.Open(INPUT_FILE, ReadOnly=True)
            time.sleep(1)
            src.Repaginate()

            # Select the page range using GoTo
            # Go to start page
            rng_start = src.GoTo(
                What=1,  # wdGoToPage
                Which=1,  # wdGoToAbsolute
                Count=start_page
            )

            # Go to end: either next page after last, or end of document
            if end_page < total_pages:
                rng_end = src.GoTo(
                    What=1,  # wdGoToPage
                    Which=1,  # wdGoToAbsolute
                    Count=end_page + 1
                )
                # Move back one character to not include the next page's content
                rng_end.MoveStart(1, -1)  # wdCharacter = 1
            else:
                rng_end = src.Content
                rng_end.Collapse(0)  # wdCollapseEnd

            # Create range from start to end
            selected_range = src.Range(rng_start.Start, rng_end.End)

            # Copy to clipboard
            selected_range.Copy()

            # Create new document and paste
            new_doc = word.Documents.Add()
            time.sleep(0.5)

            # Paste with source formatting
            new_doc.Content.PasteAndFormat(16)  # wdFormatOriginalFormatting = 16
            time.sleep(0.5)

            # Remove the initial empty paragraph if it exists before pasted content
            if new_doc.Paragraphs.Count > 0:
                first_para = new_doc.Paragraphs(1)
                if first_para.Range.Start == 0 and first_para.Range.End == 1 and first_para.Range.Text == "\r":
                    # Check if there's an empty paragraph at the very start
                    pass  # Don't delete - it might be needed

            # Save as docx (wdFormatXMLDocument = 12)
            new_doc.SaveAs2(output_path, FileFormat=12)
            new_doc.Close(0)  # wdDoNotSaveChanges
            src.Close(0)

            print(f"  Saved: {output_path}")

        print(f"\nDone! Created {num_parts} parts in {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        word.Quit(0)


if __name__ == "__main__":
    split_docx()
