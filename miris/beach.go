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

// Count cars in a frame (returns the count)
func countCars(frameDetections []Detection) int {
	count := 0
	for _, det := range frameDetections {
		if det.ClassName == "car" { // Assuming vehicle class name is "car", adjust as needed
			count++
		}
	}
	return count
}

// Check if a frame contains any person (at least 1)
func hasAnyPerson(frameDetections []Detection) bool {
	for _, det := range frameDetections {
		if det.ClassName == "people" {
			return true
		}
	}
	return false
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
		fmt.Println("Usage: go run find_frames_with_4cars_and_person.go <baseline.json file path>")
		fmt.Println("Example: go run find_frames_with_4cars_and_person.go /home/sorgenfrei/miris/data/beach/json/baseline.json")
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

		// Filter frames that have both 4 cars and person
		frameCount := 0
		for _, frameDetections := range baselineDetections {
			// Both conditions: car count >= 4 AND at least 1 person exists
			if countCars(frameDetections) >= 4 && hasAnyPerson(frameDetections) {
				frameCount++
			}
		}

		// End timing: convert microseconds to milliseconds and keep 3 decimal places
		queryDurationUs := time.Since(startTime).Microseconds()
		queryDurationMs := float64(queryDurationUs) / 1000.0

		// Accumulate statistics
		totalQueryTimeMs += queryDurationMs
		totalFrameCount += frameCount

		// Add millisecond time with 3 decimal places to CSV
		csvData = append(csvData, []string{
			fmt.Sprintf("%.3f", queryDurationMs),
		})

		// Print iteration progress
		fmt.Printf("Iteration %d/%d completed, time: %.3f ms, found matching frames (4 cars and person): %d\n",
			i+1, iterations, queryDurationMs, frameCount)
	}

	// Calculate average
	averageTimeMs := totalQueryTimeMs / float64(iterations)
	averageFrameCount := totalFrameCount / iterations

	// Output final statistics
	fmt.Println("\n===== Final Statistics =====")
	fmt.Printf("Total queries executed: %d\n", iterations)
	fmt.Printf("Average query time: %.3f ms\n", averageTimeMs)
	fmt.Printf("Average matching frames per query (4 cars and person): %d frames\n", averageFrameCount)

	// Generate CSV filename (with timestamp to prevent overwriting)
	timestamp := time.Now().Format("20060102150405")
	csvFilename := fmt.Sprintf("query_times_%s.csv", timestamp)

	// Write to CSV file
	if err := writeResultsToCSV(csvData, csvFilename); err != nil {
		fmt.Printf("Failed to write results to CSV file: %v\n", err)
	} else {
		fmt.Printf("Query time successfully exported to %s\n", csvFilename)
	}
}