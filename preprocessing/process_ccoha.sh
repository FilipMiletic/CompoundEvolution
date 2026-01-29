# List of noun compounds; can be TSV file with the compounds in the first column
targets_file=

# Input directory containing one zip file per decade
in_dir=

# Output directory for one-sentence-per-line corpus
out_dir=

for year in {1830..2000..10}; do
    in_file=${in_dir}/cleaned_${year}s.zip
    python3 process_ccoha.py $targets_file $in_file $out_dir --pos
done
