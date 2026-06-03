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

// Check if a frame contains people walking side by side
// Side-by-side logic: at least 2 persons, and exists a pair where "vertical height overlap >= 50% + horizontal gap <= 1/2 person width"
func hasPeopleWalkingSideBySide(frameDetections []Detection) bool {
	// 1. First filter all persons in current frame
	var persons []Detection
	for _, det := range frameDetections {
		if det.ClassName == "person" {
			persons = append(persons, det)
		}
	}
	// Need at least 2 persons to possibly walk side by side, return false if less
	if len(persons) < 2 {
		return false
	}

	// 2. Iterate through all person pairs, check if any are side by side
	for i := 0; i < len(persons); i++ {
		for j := i + 1; j < len(persons); j++ {
			p1 := persons[i]
			p2 := persons[j]

			// Calculate vertical height overlap rate (check if at same horizontal height)
			overlapTop := max(p1.Top, p2.Top)
			overlapBottom := min(p1.Bottom, p2.Bottom)
			overlapHeight := overlapBottom - overlapTop
			if overlapHeight <= 0 {
				continue // No vertical overlap, not at same height, skip
			}
			// Use smaller person height as baseline, overlap rate >= 50% means "same horizontal height"
			minHeight := min(p1.Bottom-p1.Top, p2.Bottom-p2.Top)
			heightOverlapRate := float64(overlapHeight) / float64(minHeight)
			if heightOverlapRate < 0.5 {
				continue
			}

			// Calculate horizontal gap (check if "side by side and close")
			var horizontalGap int
			if p1.Left < p2.Left {
				horizontalGap = p2.Left - p1.Right
			} else {
				horizontalGap = p1.Left - p2.Right
			}
			// Use smaller person width as baseline, gap <= 1/2 width means "close and side by side"
			minWidth := min(p1.Right-p1.Left, p2.Right-p2.Left)
			if horizontalGap <= minWidth/2 {
				return true // Meets side-by-side condition, return directly
			}
		}
	}

	// After iterating all pairs, no side-by-side found, return false
	return false
}

// Check if a frame contains elec (assuming elec is the class name, adjust as needed)
func hasElec(frameDetections []Detection) bool {
	for _, det := range frameDetections {
		if det.ClassName == "elec" { // Detected elec class
			return true
		}
	}
	return false
}

// Helper function: get maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Helper function: get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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
		fmt.Println("Usage: go run find_frames_with_side_by_side_people_and_elec.go <baseline.json file path>")
		fmt.Println("Example: go run find_frames_with_side_by_side_people_and_elec.go /home/sorgenfrei/miris/data/beach/json/baseline.json")
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

		// Filter frames with both people walking side by side and elec
		frameCount := 0
		for _, frameDetections := range baselineDetections {
			// Both conditions must be met
			if hasPeopleWalkingSideBySide(frameDetections) && hasElec(frameDetections) {
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
		fmt.Printf("Iteration %d/%d completed, time: %.3f ms, found matching frames (people side by side and elec): %d\n",
			i+1, iterations, queryDurationMs, frameCount)
	}

	// Calculate average
	averageTimeMs := totalQueryTimeMs / float64(iterations)
	averageFrameCount := totalFrameCount / iterations

	// Output final statistics
	fmt.Println("\n===== Final Statistics =====")
	fmt.Printf("Total queries executed: %d\n", iterations)
	fmt.Printf("Average query time: %.3f ms\n", averageTimeMs)
	fmt.Printf("Average matching frames per query (people side by side and elec): %d frames\n", averageFrameCount)

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