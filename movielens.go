package main

import (
	"bufio"
	"encoding/csv"
	"flag" 
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// --- CAMBIO: Estructura de Mapa Seguro con Mutex ---
// Esto reemplaza la necesidad de un 'collector'
type SafeSimilarityMap struct {
	mu   sync.Mutex // El "candado"
	data map[string][]SimilarityScore
}

// Método para agregar un resultado de forma segura (concurrente)
func (s *SafeSimilarityMap) Add(result Result) {
	s.mu.Lock() // <-- Poner el candado
	// Esta sección de código está protegida.
	s.data[result.UserA_ID] = append(s.data[result.UserA_ID], SimilarityScore{UserID: result.UserB_ID, Score: result.Score})
	s.data[result.UserB_ID] = append(s.data[result.UserB_ID], SimilarityScore{UserID: result.UserA_ID, Score: result.Score})
	s.mu.Unlock() // <-- Quitar el candado
}

// Estructuras de datos para MovieLens
type UserRatings map[string]map[string]float64
type MovieData map[string]string

// Estructuras para resultados
type SimilarityScore struct {
	UserID string
	Score  float64
}
type Recommendation struct {
	Game  string
	Score int
}

// CAMBIO: Struct de resultados con más métricas
type RunResult struct {
	SampleSize      int
	NumWorkers      int           // MÉTRICA NUEVA
	DurationLoad    time.Duration // MÉTRICA NUEVA
	DurationCalc    time.Duration // MÉTRICA NUEVA
	DurationRec     time.Duration // MÉTRICA NUEVA
	DurationTotal   time.Duration
	TargetUserID    string
	Recommendations []string
}

// Estructuras del Pipeline (Job se mantiene)
type Job struct {
	UserA_ID string
	UserB_ID string
}
type Result struct { // Usado solo para pasar datos al método Add
	UserA_ID string
	UserB_ID string
	Score    float64
}

// --- CAMBIO: Funciones de carga ahora devuelven el tiempo ---
func loadRatings(filePath string) (UserRatings, []string, time.Duration) {
	start := time.Now()
	fmt.Println("Iniciando la carga de ratings (ratings.dat)...")
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Error al abrir el archivo %s: %v", filePath, err)
	}
	defer file.Close()

	userRatings := make(UserRatings)
	userSet := make(map[string]bool)
	scanner := bufio.NewScanner(file)
	var lineCount int64

	fmt.Print("Procesando archivo: ")
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "::")
		if len(parts) < 3 {
			continue
		}

		lineCount++
		if lineCount%1000000 == 0 {
			fmt.Print(".")
		}

		userID, movieID, ratingStr := parts[0], parts[1], parts[2]
		rating, err := strconv.ParseFloat(ratingStr, 64)
		if err != nil {
			continue
		}

		if _, ok := userRatings[userID]; !ok {
			userRatings[userID] = make(map[string]float64)
		}
		userRatings[userID][movieID] = rating
		userSet[userID] = true
	}
	fmt.Println("\n¡Hecho!")

	userIDs := make([]string, 0, len(userSet))
	for userID := range userSet {
		userIDs = append(userIDs, userID)
	}
	duration := time.Since(start)
	fmt.Printf("Carga de ratings completada (%d usuarios) en %v.\n", len(userIDs), duration)
	return userRatings, userIDs, duration
}

func loadMovies(filePath string) (MovieData, time.Duration) {
	start := time.Now()
	fmt.Println("Iniciando la carga de películas (movies.dat)...")
	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Error al abrir el archivo %s: %v", filePath, err)
	}
	defer file.Close()

	movieData := make(MovieData)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "::")
		if len(parts) < 2 {
			continue
		}
		movieID, title := parts[0], parts[1]
		movieData[movieID] = title
	}
	duration := time.Since(start)
	fmt.Printf("Carga de películas completada (%d películas) en %v.\n", len(movieData), duration)
	return movieData, duration
}

// Lógica de Similitud (sin cambios)
func cosineSimilarity(ratingsA, ratingsB map[string]float64) float64 {
	dotProduct := 0.0
	normA_sq := 0.0
	normB_sq := 0.0

	for movieID, ratingA := range ratingsA {
		if ratingB, ok := ratingsB[movieID]; ok {
			dotProduct += ratingA * ratingB
		}
		normA_sq += ratingA * ratingA
	}

	for _, ratingB := range ratingsB {
		normB_sq += ratingB * ratingB
	}

	if normA_sq == 0 || normB_sq == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA_sq) * math.Sqrt(normB_sq))
}

// Lógica de Recomendación (sin cambios)
func generateRecommendations(targetUserID string, similarityMap map[string][]SimilarityScore, userRatings UserRatings, n int, k int) ([]Recommendation, time.Duration) {
	start := time.Now()
	targetSimilarities, ok := similarityMap[targetUserID]
	if !ok {
		return nil, time.Since(start)
	}
	sort.Slice(targetSimilarities, func(i, j int) bool {
		return targetSimilarities[i].Score > targetSimilarities[j].Score
	})
	neighbors := targetSimilarities
	if len(neighbors) > n {
		neighbors = neighbors[:n]
	}
	candidateGames := make(map[string]int)
	targetUserRatings := userRatings[targetUserID]

	for _, neighbor := range neighbors {
		for movieID := range userRatings[neighbor.UserID] {
			if _, ok := targetUserRatings[movieID]; !ok {
				candidateGames[movieID]++
			}
		}
	}

	recommendations := make([]Recommendation, 0, len(candidateGames))
	for movieID, score := range candidateGames {
		recommendations = append(recommendations, Recommendation{Game: movieID, Score: score})
	}

	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].Score > recommendations[j].Score
	})
	if len(recommendations) > k {
		return recommendations[:k], time.Since(start)
	}
	return recommendations, time.Since(start)
}

// COMPONENTES DEL PIPELINE CONCURRENTE
// (Generator no tiene cambios)
func generator(userIDs []string, jobs chan<- Job) {
	defer close(jobs)
	for i := 0; i < len(userIDs); i++ {
		for j := i + 1; j < len(userIDs); j++ {
			jobs <- Job{UserA_ID: userIDs[i], UserB_ID: userIDs[j]}
		}
	}
	fmt.Println("\n¡Generator terminó de crear todos los jobs!")
}

// --- CAMBIO: 'worker' ahora escribe en el Mapa Seguro ---
// Ya no necesita el canal 'results'
func worker(userRatings UserRatings, jobs <-chan Job, safeMap *SafeSimilarityMap, wg *sync.WaitGroup) {
	defer wg.Done()
	for job := range jobs {
		ratingsA := userRatings[job.UserA_ID]
		ratingsB := userRatings[job.UserB_ID]
		score := cosineSimilarity(ratingsA, ratingsB)

		if score > 0 {
			// Escribimos directamente en el mapa protegido
			result := Result{UserA_ID: job.UserA_ID, UserB_ID: job.UserB_ID, Score: score}
			safeMap.Add(result)
		}
	}
}

// --- CAMBIO: El 'collector' se ha ELIMINADO ---

// --- CAMBIO: Función principal del pipeline ---
// Ahora crea el SafeSimilarityMap y no usa 'results' ni 'collector'
func calculateAllSimilaritiesConcurrent(userRatings UserRatings, userIDs []string, numWorkers int) (map[string][]SimilarityScore, time.Duration) {
	start := time.Now()
	jobs := make(chan Job, 1000) // El buffer de jobs se mantiene
	var wg sync.WaitGroup

	// 1. Creamos nuestro Mapa Seguro
	safeMap := SafeSimilarityMap{
		data: make(map[string][]SimilarityScore),
	}

	// 2. Iniciamos el Generator
	go generator(userIDs, jobs)

	// 3. Iniciamos los Workers
	fmt.Printf("Iniciando %d workers para el cálculo...\n", numWorkers)
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		// Pasamos el 'safeMap' al worker
		go worker(userRatings, jobs, &safeMap, &wg)
	}

	// 4. Esperamos que los workers terminen
	// (El generator cierra 'jobs', los workers se vacían y terminan)
	wg.Wait()
	fmt.Println("¡Todos los workers han terminado!")

	duration := time.Since(start)
	// 5. Devolvemos el mapa interno
	return safeMap.data, duration
}

// --- CAMBIO: writeResultsToCSV actualizado con las nuevas métricas ---
func writeResultsToCSV(results []RunResult, filename string) {
	fmt.Printf("\nGuardando resultados en el archivo %s...\n", filename)
	file, err := os.Create(filename)
	if err != nil {
		log.Fatalf("No se pudo crear el archivo CSV: %v", err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Nuevos encabezados
	header := []string{
		"SampleSize",
		"NumWorkers",
		"TiempoCarga_s",
		"TiempoCalculo_s",
		"TiempoRecomendacion_s",
		"TiempoTotal_s",
		"TargetUserID",
		"Recommendations",
	}
	writer.Write(header)

	for _, result := range results {
		recString := strings.Join(result.Recommendations, "|")
		if recString == "" {
			recString = "N/A"
		}
		row := []string{
			strconv.Itoa(result.SampleSize),
			strconv.Itoa(result.NumWorkers),
			fmt.Sprintf("%f", result.DurationLoad.Seconds()),
			fmt.Sprintf("%f", result.DurationCalc.Seconds()),
			fmt.Sprintf("%f", result.DurationRec.Seconds()),
			fmt.Sprintf("%f", result.DurationTotal.Seconds()),
			result.TargetUserID,
			recString,
		}
		writer.Write(row)
	}
	fmt.Println("Resultados guardados exitosamente.")
}

// --- CAMBIO: 'main' actualizado con flags y nuevas métricas ---
// --- CAMBIO: 'main' actualizado para testear múltiples workers ---
func main() {
	// --- 1. Configuración con Flags ---
	// Definimos los flags que el usuario puede pasar
	sampleSizeFlag := flag.Int("size", 0, "Número de registros a procesar. 0 para usar el dataset entero.")
	outputFileFlag := flag.String("output", "benchmark", "Prefijo para los archivos CSV de salida (ej. 'benchmark' -> 'benchmark_8w.csv').")
	
	// El flag de worker se elimina
	flag.Parse() // Leemos los flags

	// --- NUEVO: Arreglo de Conteo de Workers a Probar ---
	// Puedes modificar este arreglo con los valores que quieras testear
	workerCounts := []int{4, 12, 20} 

	// --- 2. Carga de Datos (Se hace UNA SOLA VEZ) ---
	startLoad := time.Now()
	// Asegúrate de que las rutas a tus archivos .dat sean correctas
	movieData, _ := loadMovies("./ml-10M100K/movies.dat")
	userRatings, userIDs, _ := loadRatings("./ml-10M100K/ratings.dat")
	durationLoad := time.Since(startLoad)

	if len(userIDs) == 0 {
		log.Fatal("No se encontraron usuarios en el dataset. Saliendo.")
	}

	// --- 3. Selección de Muestra (Sample) (Se hace UNA SOLA VEZ) ---
	var sampleUserIDs []string
	sampleSize := *sampleSizeFlag

	if sampleSize <= 0 || sampleSize > len(userIDs) {
		fmt.Printf("\nUsando el dataset completo (%d usuarios).\n", len(userIDs))
		sampleSize = len(userIDs)
		sampleUserIDs = userIDs
	} else {
		fmt.Printf("\nUsando una muestra de %d usuarios.\n", sampleSize)
		// Barajar la lista para obtener una muestra aleatoria
		r := rand.New(rand.NewSource(42))
		r.Shuffle(len(userIDs), func(i, j int) {
			userIDs[i], userIDs[j] = userIDs[j], userIDs[i]
		})
		sampleUserIDs = userIDs[:sampleSize]
	}

	// 4. Seleccionar Usuario Objetivo (Se hace UNA SOLA VEZ)
	var targetUser string
	maxRatings := -1
	for _, userID := range sampleUserIDs {
		if len(userRatings[userID]) > maxRatings {
			maxRatings = len(userRatings[userID])
			targetUser = userID
		}
	}
	fmt.Printf("Usuario objetivo seleccionado (con %d ratings): %s\n", maxRatings, targetUser)


	// --- 5. Bucle de Ejecución del Benchmark ---
	fmt.Println("\n=============================================")
	fmt.Printf(" INICIANDO PRUEBAS DE BENCHMARK (Size=%d)\n", sampleSize)
	fmt.Println("=============================================")

	var allResults []RunResult // Para guardar todos los resultados para el resumen final

	// Bucle sobre cada conteo de workers
	for _, numWorkers := range workerCounts {
		
		fmt.Printf("\n--- Iniciando prueba con %d WORKERS ---\n", numWorkers)

		// --- LLAMADAS A LAS FUNCIONES CON TIMING ---
		similarityMap, durationCalc := calculateAllSimilaritiesConcurrent(userRatings, sampleUserIDs, numWorkers)
		recommendations, durationRec := generateRecommendations(targetUser, similarityMap, userRatings, 10, 7)

		durationTotal := durationLoad + durationCalc + durationRec

		// Convertir MovieIDs a Títulos
		recNames := make([]string, len(recommendations))
		for i, rec := range recommendations {
			recNames[i] = movieData[rec.Game]
		}

		fmt.Printf("\nTiempo total (cálculo + recomendación) para %d workers: %v\n", numWorkers, durationTotal)

		// Guardar el resultado de esta corrida
		result := RunResult{
			SampleSize:      sampleSize,
			NumWorkers:      numWorkers,
			DurationLoad:    durationLoad,
			DurationCalc:    durationCalc,
			DurationRec:     durationRec,
			DurationTotal:   durationTotal,
			TargetUserID:    targetUser,
			Recommendations: recNames,
		}
		allResults = append(allResults, result) // Guardamos para el resumen final

		// --- 6. Resumen y Guardado (DENTRO DEL BUCLE) ---
		fmt.Println("\n-------------------------------------------------------------------------------------------------------")
		fmt.Printf("                          RESUMEN DE LA EJECUCIÓN (Workers: %d)\n", numWorkers)
		fmt.Println("-------------------------------------------------------------------------------------------------------")
		fmt.Printf("%-20s | %-15s\n", "Métrica", "Tiempo (s)")
		fmt.Println("----------------------|-----------------")
		fmt.Printf("%-20s | %-15.4f\n", "Tiempo de Carga", result.DurationLoad.Seconds())
		fmt.Printf("%-20s | %-15.4f\n", "Tiempo de Cálculo (C)", result.DurationCalc.Seconds())
		fmt.Printf("%-20s | %-15.4f\n", "Tiempo de Recomend. (S)", result.DurationRec.Seconds())
		fmt.Printf("%-20s | %-15.4f\n", "TIEMPO TOTAL", result.DurationTotal.Seconds())
		fmt.Println("-------------------------------------------------------------------------------------------------------")

		// Generar nombre de archivo dinámico
		outputFilename := fmt.Sprintf("%s_%dw.csv", *outputFileFlag, numWorkers)
		
		// Escribir en un CSV *separado* para esta prueba
		writeResultsToCSV([]RunResult{result}, outputFilename)
	}

	// --- 7. Resumen Final de Todas las Pruebas (FUERA DEL BUCLE) ---
	fmt.Println("\n=======================================================================================================")
	fmt.Printf("                       RESUMEN FINAL DE BENCHMARK (SampleSize: %d)\n", sampleSize)
	fmt.Println("=======================================================================================================")
	fmt.Printf("%-15s | %-20s | %-20s | %-20s\n", "Num Workers", "Tiempo Cálculo (s)", "Tiempo Recomendación (s)", "Tiempo Total (s)")
	fmt.Println("----------------|----------------------|------------------------|----------------------")
	for _, res := range allResults {
		fmt.Printf("%-15d | %-20.4f | %-20.4f | %-20.4f\n", 
			res.NumWorkers, 
			res.DurationCalc.Seconds(), 
			res.DurationRec.Seconds(),
			res.DurationTotal.Seconds())
	}
	fmt.Println("=======================================================================================================")
}