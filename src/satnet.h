typedef struct mix_t {
    int b, n, m, k;
    int32_t *is_input;  // b*n
    int32_t *index;     // b*n
    int32_t *niter;     // b
    float *S, *dS;      // n*m
    float *z, *dz;      // b*n
    float *V, *U;       // b*n*k
    float *W, *Phi;     // b*m*k
    float *gnrm, *Snrms;// b*n
    float *cache;
} mix_t ;
