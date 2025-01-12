#!/usr/bin/env bash

# Define the list of metrics
metrics=(
    "gpu__time_duration.sum"
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "lts__t_sectors_srcunit_tex_op_read.sum"
    "lts__t_sectors_srcunit_tex_op_write.sum"
    "lts__t_sector_hit_rate.pct"
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
    "smsp__inst_executed.sum"
)

# Iterate over each .ncu-rep file in the current directory
for repfile in *.ncu-rep; do

    # Create an output CSV file named "<basename>_metrics.csv"
    csvfile="${repfile%.ncu-rep}_metrics.csv"

    # Remove the CSV file if it already exists
    [ -f "$csvfile" ] && rm -f "$csvfile"

    # Write a single header row to the CSV file
    echo "Kernel Name,Block Size,Grid Size,metric,Value" >> "$csvfile"

    # For each metric, collect data and parse/append to the CSV
    for metric in "${metrics[@]}"; do
        # Run Nsight Compute in CSV mode and store its output
        out=$(ncu -i "$repfile" --page raw --metrics "$metric" --csv)

        # Parse the CSV data. We skip:
        #   - line 1 (the CSV header)
        #   - line 2 (the CSV units)
        # Then for each subsequent line, we extract columns:
        #   $5  -> Kernel Name
        #   $8  -> Block Size
        #   $9  -> Grid Size
        #   $NF -> The last column’s value (the metric value for this row).
        # We place "metric" (the metric’s name) into a new column.
        echo "$out" | awk -v metricName="$metric" '
            BEGIN {
                FS="\",\""
                OFS=","
            }
            NR==1 { next }       # Skip the header line
            NR==2 { next }       # Skip the units line
            {
                # Remove leading/trailing quotes from columns of interest
                sub(/^"/,"",$5);  sub(/"$/,"",$5)
                sub(/^"/,"",$8);  sub(/"$/,"",$8)
                sub(/^"/,"",$9);  sub(/"$/,"",$9)
                sub(/^"/,"",$NF); sub(/"$/,"",$NF)

                kernelName=$5
                blockSize=$8
                gridSize=$9
                value=$NF

                # Print the desired columns in our final CSV format
                print kernelName, blockSize, gridSize, metricName, value
            }
        ' >> "$csvfile"
    done

    echo "Metrics for $repfile appended to $csvfile."
done
