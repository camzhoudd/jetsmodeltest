#!/usr/bin/env python3
import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple


def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
	with open(csv_path, 'r', newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		return list(reader)


def safe_json_loads(text: str):
	if text is None:
		return None
	text = text.strip()
	if not text:
		return None
	try:
		return json.loads(text)
	except json.JSONDecodeError as e:
		logging.warning("Failed to parse JSON; length=%s error=%s", len(text), e)
		return None


def download_bytes(url: str, max_retries: int = 3, timeout: int = 60) -> Optional[bytes]:
	last_err: Optional[Exception] = None
	headers = {
		'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
		'Accept': '*/*',
	}
	for attempt in range(1, max_retries + 1):
		try:
			req = urllib.request.Request(url, headers=headers)
			with urllib.request.urlopen(req, timeout=timeout) as resp:
				return resp.read()
		except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
			last_err = e
			wait_s = min(2 ** attempt, 10)
			logging.warning("Download failed (attempt %d/%d) for %s: %s", attempt, max_retries, url, e)
			time.sleep(wait_s)
	if last_err is not None:
		logging.error("Giving up downloading %s: %s", url, last_err)
	return None


def extract_zip_entries(zip_bytes: bytes) -> Dict[str, bytes]:
	result: Dict[str, bytes] = {}
	with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
		for zi in zf.infolist():
			# Normalize to just the base filename (strip directories)
			base_name = os.path.basename(zi.filename)
			with zf.open(zi, 'r') as fp:
				result[base_name] = fp.read()
	return result


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def parse_int_suffix_from_filename(filename: str) -> Optional[int]:
	match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
	if not match:
		return None
	return int(match.group(1))


def normalize_field_name(name: str) -> str:
	"""Normalize CSV header field names by removing whitespace characters."""
	return re.sub(r"\s+", "", name or "")


def get_field_value(row: Dict[str, str], desired_field: str) -> Optional[str]:
	"""Fetch a field value from a CSV row, tolerant to whitespace in header names."""
	if desired_field in row:
		return row.get(desired_field)
	desired_norm = normalize_field_name(desired_field)
	for k, v in row.items():
		if normalize_field_name(k) == desired_norm:
			return v
	return None


def is_all_zeros(s: str) -> bool:
	return bool(re.fullmatch(r"0+", s))


def process_shelf_tags_row(
	row: Dict[str, str],
	zip_url_field: str,
	json_field: str,
	image_counter_start: int,
	output_dir: str,
	metadata_file_path: str,
) -> int:
	"""
	Processes a single CSV row. Returns the next image counter value after processing this row.
	"""
	zip_url = (get_field_value(row, zip_url_field) or '').strip()
	json_blob = get_field_value(row, json_field)
	items = safe_json_loads(json_blob)

	if not zip_url:
		logging.info("Skipping row without %s", zip_url_field)
		return image_counter_start
	if not isinstance(items, list) or not items:
		logging.info("Skipping row with empty or invalid %s", json_field)
		return image_counter_start

	zip_bytes = download_bytes(zip_url)
	if not zip_bytes:
		logging.error("No data for zip at %s", zip_url)
		return image_counter_start

	zip_entries = extract_zip_entries(zip_bytes)
	if not zip_entries:
		logging.warning("Zip at %s contained no entries", zip_url)
		return image_counter_start

	# Build mapping from filename -> (barcode, source), filtering out missing/zero barcodes
	filename_to_meta: Dict[str, Tuple[str, str]] = {}
	for obj in items:
		if not isinstance(obj, dict):
			continue
		uploaded = obj.get('uploaded_image') or {}
		filename = (uploaded.get('image_filename') or '').strip()
		if not filename:
			continue
		barcode_raw = obj.get('barcode')
		barcode = ('' if barcode_raw in (None, 'null') else str(barcode_raw)).strip()
		if not barcode or is_all_zeros(barcode):
			# Skip if no barcode or barcode is all zeros
			continue
		source_raw = obj.get('barcode_detection_source')
		source = (source_raw if source_raw not in (None, 'null') else '').strip()
		filename_to_meta[filename] = (barcode, source)

	# Iterate metadata items in a deterministic order: by numeric suffix if present, else by name
	def sort_key(name: str) -> Tuple[int, str]:
		idx = parse_int_suffix_from_filename(name)
		return ((idx if idx is not None else 10**9), name)

	sorted_filenames = sorted(filename_to_meta.keys(), key=sort_key)

	counter = image_counter_start
	with open(metadata_file_path, 'a', encoding='utf-8') as meta_fp:
		for filename in sorted_filenames:
			if filename not in zip_entries:
				logging.warning("Filename %s listed in metadata but missing in zip", filename)
				continue
			image_bytes = zip_entries[filename]
			barcode, source = filename_to_meta.get(filename, ('', ''))
			# Save image and write metadata line
			counter += 1
			new_name = f"{counter}.heif"
			out_path = os.path.join(output_dir, new_name)
			with open(out_path, 'wb') as img_fp:
				img_fp.write(image_bytes)
			meta_fp.write(f"{barcode}, {source}\n")
	return counter


def main():
	parser = argparse.ArgumentParser(description='Extract shelf tag images and metadata from CSV')
	parser.add_argument('--csv', required=True, help='Path to the CSV file')
	parser.add_argument('--out', required=True, help='Output directory to save images and metadata.txt')
	parser.add_argument('--start', type=int, default=0, help='Starting index for image numbering (default: 0)')
	parser.add_argument('--zip-field', default='SHELF_TAGS', help='CSV column name for the zip URL (default: SHELF_TAGS)')
	parser.add_argument('--json-field', default='SHELF_TAGS_IMAGES_URL', help='CSV column name for JSON metadata (default: SHELF_TAGS_IMAGES_URL)')
	parser.add_argument('--limit-rows', type=int, default=0, help='If >0, process only this many rows')
	parser.add_argument('--log-level', default='INFO', help='Logging level (e.g., INFO, DEBUG)')
	args = parser.parse_args()

	logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(levelname)s: %(message)s')

	ensure_dir(args.out)
	metadata_file_path = os.path.join(args.out, 'metadata.txt')

	rows = read_csv_rows(args.csv)
	if not rows:
		logging.warning("No rows found in CSV")
		return
	logging.info("Loaded %d rows", len(rows))

	image_counter = args.start
	processed_rows = 0
	for idx, row in enumerate(rows):
		if args.limit_rows and processed_rows >= args.limit_rows:
			break
		try:
			image_counter = process_shelf_tags_row(
				row=row,
				zip_url_field=args.zip_field,
				json_field=args.json_field,
				image_counter_start=image_counter,
				output_dir=args.out,
				metadata_file_path=metadata_file_path,
			)
			processed_rows += 1
		except Exception as e:
			logging.exception("Error processing row %d: %s", idx, e)

	logging.info("Done. Saved images up to index %d in %s and metadata in %s", image_counter, args.out, metadata_file_path)


if __name__ == '__main__':
	main() 