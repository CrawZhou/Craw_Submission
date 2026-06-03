package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Detection struct matching baseline.json structure
type Detection struct {
	FrameIdx  int    `json:"frame_idx"`  // Frame index
	TrackID   int    `json:"track_id"`   // Track ID
	Left      int    `json:"left"`       // Left boundary
	Top       int    `json:"top"`        // Top boundary
	Right     int    `json:"right"`      // Right boundary
	Bottom    int    `json:"bottom"`     // Bottom boundary
	ClassID   int    `json:"class_id"`   // Class ID
	ClassName string `json:"class_name"` // Class name
}

// Core filter function: check if a frame contains both car and van (new van check logic)
func hasBothCarAndVan(frameDetections []Detection) bool {
	hasCar := false
	hasVan := false

	// Iterate through all detections, check if both car and van exist
	for _, det := range frameDetections {
		switch det.ClassName {
		case "car":
			hasCar = true
		case "van":
			hasVan = true
		}

		// Early exit: when both car and van are found, stop iterating
		if hasCar && hasVan {
			return true
		}
	}

	// Need both car and van to return true
	return hasCar && hasVan
}

// Read JSON file
func readJSON(filePath string, v interface{}) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(v); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}
	return nil
}

// Write query time data to CSV file
func writeResultsToCSV(results [][]string, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write all rows
	for _, row := range results {
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV: %w", err)
		}
	}

	return nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run find_frames_with_more_than_10_cars.go <baseline.json file path>")
		fmt.Println("Example: go run find_frames_with_more_than_10_cars.go /home/sorgenfrei/miris/data/beach/json/baseline.json")
		os.Exit(1)
	}
	baselinePath := os.Args[1]
	const iterations = 200

	var totalQueryTimeMs float64 = 0 // Use float64 to store millisecond-level total time, preserving precision
	var totalFrameCount int = 0

	// Keep CSV header as "Query Time (milliseconds)"
	csvData := [][]string{
		{"Query Time (milliseconds)"},
	}

	for i := 0; i < iterations; i++ {
		// Start timing - includes file reading and filtering process
		startTime := time.Now()

		// Read baseline.json (included in query time)
		var baselineDetections [][]Detection
		if err := readJSON(baselinePath, &baselineDetections); err != nil {
			panic(fmt.Sprintf("failed to read baseline.json: %v", err))
		}

		// Filter frames with both car and van (call the new filter function)
		frameCount := 0
		for _, frameDetections := range baselineDetections {
			if hasBothCarAndVan(frameDetections) { // Replace original hasAnyCar function
				frameCount++
			}
		}

		// End timing: convert microseconds to milliseconds and keep 3 decimal places (logic unchanged)
		queryDurationUs := time.Since(startTime).Microseconds()
		queryDurationMs := float64(queryDurationUs) / 1000.0

		// Accumulate statistics (logic unchanged)
		totalQueryTimeMs += queryDurationMs
		totalFrameCount += frameCount

		// Add millisecond time with 3 decimal places to CSV (logic unchanged)
		csvData = append(csvData, []string{
			fmt.Sprintf("%.3f", queryDurationMs),
		})

		// Print iteration progress, text updated to "frames with both car and van"
		fmt.Printf("Iteration %d/%d completed, time: %.3f ms, found matching frames (car and van): %d\n",
			i+1, iterations, queryDurationMs, frameCount)
	}

	// Calculate average (logic unchanged)
	averageTimeMs := totalQueryTimeMs / float64(iterations)
	averageFrameCount := totalFrameCount / iterations

	// Output final statistics, text updated to "frames with both car and van"
	fmt.Println("\n===== Final Statistics =====")
	fmt.Printf("Total queries executed: %d\n", iterations)
	fmt.Printf("Average query time: %.3f ms\n", averageTimeMs)
	fmt.Printf("Average matching frames per query (car and van): %d frames\n", averageFrameCount)

	// Generate CSV filename (logic unchanged, with timestamp to prevent overwriting)
	timestamp := time.Now().Format("20060102150405")
	csvFilename := fmt.Sprintf("query_times_%s.csv", timestamp)

	// Write to CSV file (logic unchanged)
	if err := writeResultsToCSV(csvData, csvFilename); err != nil {
		fmt.Printf("Failed to write results to CSV file: %v\n", err)
	} else {
		fmt.Printf("Query time successfully exported to %s\n", csvFilename)
	}
}