/**
 * This is a utility funstion for checking CUDA Errors,
 * NOTE: This function was taken from a blog www.beechwood.eu
 * @param err
 * @param file
 * @param line
 */
static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    int aa = 0;
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    scanf("%d", &aa);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

