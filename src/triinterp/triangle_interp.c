/**
 * Triangle interpolation with analytic gradients for CasADi external function.
 *
 * Supports hash-based loading for compiled NLP caching:
 * - Python computes SHA256 hash of interpolator data
 * - Data saved to binary file named by hash
 * - C code loads data lazily by hash lookup
 *
 * Compile with:
 *   gcc -shared -fPIC -O3 -o libtriinterp.so triangle_interp.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* CasADi types */
typedef double casadi_real;
typedef long long int casadi_int;

/* Data structure for one interpolator instance */
typedef struct {
    int n_tri;
    int n_points;
    int n_levels;

    /* Owned arrays (allocated by load_from_binary) */
    int64_t *tri_vertices;
    double *tri_y;
    double *tri_theta;
    double *tri_values;
    int64_t *strip_starts;
    double *y_levels;
} TriInterpData;

/* ============================================================
 * Hash-based cache for compiled NLP support
 * ============================================================ */

#define MAX_LOADED 64
#define MAX_PATH_LEN 512

/* Cache entry: hash -> loaded data */
static struct {
    uint64_t hash;
    TriInterpData *data;
} loaded_cache[MAX_LOADED];
static int loaded_count = 0;

/* Cache directory path (set by Python) */
static char cache_dir[MAX_PATH_LEN] = ".";

/* Set cache directory (called from Python) */
void triinterp_set_cache_dir(const char *path) {
    strncpy(cache_dir, path, MAX_PATH_LEN - 1);
    cache_dir[MAX_PATH_LEN - 1] = '\0';
}

/* Load interpolator data from binary file */
static TriInterpData* load_from_binary(uint64_t hash) {
    char path[MAX_PATH_LEN + 80];
    /* Use 13 hex chars (52 bits) to match Python's hash format */
    snprintf(path, sizeof(path), "%s/%013llx.bin", cache_dir, (unsigned long long)hash);

    FILE *f = fopen(path, "rb");
    if (!f) {
        /* Try with full 64-char hash (Python saves with full hex) */
        /* We only have 64 bits, so we need to search for matching prefix */
        /* For now, return NULL - Python ensures file exists before call */
        return NULL;
    }

    /* Read metadata: [n_tri, n_points, n_levels] as int32 */
    int32_t meta[3];
    if (fread(meta, sizeof(int32_t), 3, f) != 3) {
        fclose(f);
        return NULL;
    }
    int n_tri = meta[0];
    int n_points = meta[1];
    int n_levels = meta[2];

    /* Allocate data structure */
    TriInterpData *data = (TriInterpData*)malloc(sizeof(TriInterpData));
    if (!data) {
        fclose(f);
        return NULL;
    }
    data->n_tri = n_tri;
    data->n_points = n_points;
    data->n_levels = n_levels;

    /* Allocate and read arrays */
    data->tri_vertices = (int64_t*)malloc(n_tri * 3 * sizeof(int64_t));
    data->tri_y = (double*)malloc(n_points * sizeof(double));
    data->tri_theta = (double*)malloc(n_points * sizeof(double));
    data->tri_values = (double*)malloc(n_points * sizeof(double));
    data->strip_starts = (int64_t*)malloc(n_levels * sizeof(int64_t));
    data->y_levels = (double*)malloc(n_levels * sizeof(double));

    if (!data->tri_vertices || !data->tri_y || !data->tri_theta ||
        !data->tri_values || !data->strip_starts || !data->y_levels) {
        /* Allocation failed - clean up */
        free(data->tri_vertices);
        free(data->tri_y);
        free(data->tri_theta);
        free(data->tri_values);
        free(data->strip_starts);
        free(data->y_levels);
        free(data);
        fclose(f);
        return NULL;
    }

    /* Read arrays in order (matching Python's _save_binary) */
    size_t read_ok = 1;
    read_ok &= (fread(data->tri_vertices, sizeof(int64_t), n_tri * 3, f) == (size_t)(n_tri * 3));
    read_ok &= (fread(data->tri_y, sizeof(double), n_points, f) == (size_t)n_points);
    read_ok &= (fread(data->tri_theta, sizeof(double), n_points, f) == (size_t)n_points);
    read_ok &= (fread(data->tri_values, sizeof(double), n_points, f) == (size_t)n_points);
    read_ok &= (fread(data->strip_starts, sizeof(int64_t), n_levels, f) == (size_t)n_levels);
    read_ok &= (fread(data->y_levels, sizeof(double), n_levels, f) == (size_t)n_levels);

    fclose(f);

    if (!read_ok) {
        free(data->tri_vertices);
        free(data->tri_y);
        free(data->tri_theta);
        free(data->tri_values);
        free(data->strip_starts);
        free(data->y_levels);
        free(data);
        return NULL;
    }

    return data;
}

/* Get or load interpolator by hash */
static TriInterpData* get_or_load(uint64_t hash) {
    /* Check cache first */
    for (int i = 0; i < loaded_count; i++) {
        if (loaded_cache[i].hash == hash) {
            return loaded_cache[i].data;
        }
    }

    /* Not in cache - load from binary file */
    TriInterpData *data = load_from_binary(hash);
    if (data && loaded_count < MAX_LOADED) {
        loaded_cache[loaded_count].hash = hash;
        loaded_cache[loaded_count].data = data;
        loaded_count++;
    }

    return data;
}

/**
 * Core interpolation with gradients.
 */
static void find_triangle_interp_with_grad(
    double y, double theta,
    const TriInterpData *data,
    double *value, double *dv_dy, double *dv_dtheta)
{
    int n_levels = data->n_levels;
    double *y_levels = data->y_levels;

    if (y < y_levels[0] || y > y_levels[n_levels - 1]) {
        *value = NAN;
        *dv_dy = 0.0;
        *dv_dtheta = 0.0;
        return;
    }

    int lo = 0, hi = n_levels - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (y_levels[mid] <= y) lo = mid;
        else hi = mid - 1;
    }
    int strip_idx = lo;

    int t_start = data->strip_starts[strip_idx];
    int t_end = data->strip_starts[strip_idx + 1];
    if (strip_idx > 0) t_start = data->strip_starts[strip_idx - 1];
    if (strip_idx < n_levels - 2) t_end = data->strip_starts[strip_idx + 2];

    for (int t = t_start; t < t_end; t++) {
        int64_t i0 = data->tri_vertices[t * 3 + 0];
        int64_t i1 = data->tri_vertices[t * 3 + 1];
        int64_t i2 = data->tri_vertices[t * 3 + 2];

        double y0 = data->tri_y[i0], t0 = data->tri_theta[i0];
        double y1 = data->tri_y[i1], t1 = data->tri_theta[i1];
        double y2 = data->tri_y[i2], t2 = data->tri_theta[i2];
        double v0 = data->tri_values[i0];
        double v1 = data->tri_values[i1];
        double v2 = data->tri_values[i2];

        double denom = (y1 - y2) * (t0 - t2) + (t2 - t1) * (y0 - y2);
        if (fabs(denom) < 1e-30) continue;

        double inv_denom = 1.0 / denom;
        double lam0 = ((y1 - y2) * (theta - t2) + (t2 - t1) * (y - y2)) * inv_denom;
        double lam1 = ((y2 - y0) * (theta - t2) + (t0 - t2) * (y - y2)) * inv_denom;
        double lam2 = 1.0 - lam0 - lam1;

        if (lam0 >= -1e-10 && lam1 >= -1e-10 && lam2 >= -1e-10) {
            *value = lam0 * v0 + lam1 * v1 + lam2 * v2;
            double dlam0_dy = (t2 - t1) * inv_denom;
            double dlam1_dy = (t0 - t2) * inv_denom;
            double dlam0_dt = (y1 - y2) * inv_denom;
            double dlam1_dt = (y2 - y0) * inv_denom;
            *dv_dy = v0 * dlam0_dy + v1 * dlam1_dy + v2 * (-dlam0_dy - dlam1_dy);
            *dv_dtheta = v0 * dlam0_dt + v1 * dlam1_dt + v2 * (-dlam0_dt - dlam1_dt);
            return;
        }
    }

    *value = NAN;
    *dv_dy = 0.0;
    *dv_dtheta = 0.0;
}

/* ============================================================
 * Python ctypes interface (LEGACY - for backward compatibility)
 * New code should use hash-based loading via triinterp_set_cache_dir
 * ============================================================ */

int64_t triinterp_create(
    int n_tri, int n_points, int n_levels,
    const int64_t *tri_vertices,
    const double *tri_y,
    const double *tri_theta,
    const double *tri_values,
    const int64_t *strip_starts,
    const double *y_levels)
{
    TriInterpData *data = (TriInterpData*)malloc(sizeof(TriInterpData));
    data->n_tri = n_tri;
    data->n_points = n_points;
    data->n_levels = n_levels;
    /* Note: these point to Python-owned memory (legacy mode) */
    data->tri_vertices = (int64_t*)tri_vertices;
    data->tri_y = (double*)tri_y;
    data->tri_theta = (double*)tri_theta;
    data->tri_values = (double*)tri_values;
    data->strip_starts = (int64_t*)strip_starts;
    data->y_levels = (double*)y_levels;
    return (int64_t)data;
}

void triinterp_destroy(int64_t ptr) {
    /* Note: only frees the struct, not the arrays (Python-owned in legacy mode) */
    free((void*)ptr);
}

/* ============================================================
 * CasADi external function interface
 * Matches the format generated by CasADi CodeGenerator
 * ============================================================ */

/* Sparsity pattern for scalar: {nrow, ncol, colind..., row...} */
static const casadi_int scalar_sparsity[5] = {1, 1, 0, 1, 0};

/* interp_wh: (hash_id, y, theta) -> (value)
 * First argument is now a hash ID (64-bit int as double), not a pointer.
 * Data is loaded lazily from binary cache on first use.
 */

int interp_wh(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
    uint64_t hash = (uint64_t)arg[0][0];
    TriInterpData *data = get_or_load(hash);
    if (!data) {
        if (res[0]) res[0][0] = NAN;
        return 1;  /* Error: data not found */
    }
    double value, dv_dy, dv_dt;
    find_triangle_interp_with_grad(arg[1][0], arg[2][0], data, &value, &dv_dy, &dv_dt);
    if (res[0]) res[0][0] = value;
    return 0;
}

casadi_int interp_wh_n_in(void) { return 3; }
casadi_int interp_wh_n_out(void) { return 1; }

const casadi_int* interp_wh_sparsity_in(casadi_int i) {
    return scalar_sparsity;
}

const casadi_int* interp_wh_sparsity_out(casadi_int i) {
    return scalar_sparsity;
}

int interp_wh_work(casadi_int *sz_arg, casadi_int *sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 3;
    if (sz_res) *sz_res = 1;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

/* Memory management (no-ops for stateless function) */
int interp_wh_alloc_mem(void) { return 0; }
int interp_wh_init_mem(int mem) { return 0; }
void interp_wh_free_mem(int mem) { }
int interp_wh_checkout(void) { return 0; }
void interp_wh_release(int mem) { }
void interp_wh_incref(void) { }
void interp_wh_decref(void) { }

/* ============================================================
 * Forward mode derivative: fwd1_interp_wh
 * Inputs: (ptr, y, theta, out_value, fwd_ptr, fwd_y, fwd_theta)
 * Outputs: (fwd_value)
 * ============================================================ */

/* Sparsity for unused output slot: 1x1 with 0 nonzeros */
static const casadi_int empty_sparsity[4] = {1, 1, 0, 0};

int fwd1_interp_wh(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
    uint64_t hash = (uint64_t)arg[0][0];
    TriInterpData *data = get_or_load(hash);
    if (!data) {
        if (res[0]) res[0][0] = NAN;
        return 1;
    }
    double y = arg[1][0];
    double theta = arg[2][0];
    /* arg[3] = out_value (not used) */
    /* arg[4] = fwd_hash (ignored, hash is constant) */
    double fwd_y = arg[5] ? arg[5][0] : 0;
    double fwd_theta = arg[6] ? arg[6][0] : 0;

    double value, dv_dy, dv_dt;
    find_triangle_interp_with_grad(y, theta, data, &value, &dv_dy, &dv_dt);

    if (res[0]) res[0][0] = dv_dy * fwd_y + dv_dt * fwd_theta;
    return 0;
}

casadi_int fwd1_interp_wh_n_in(void) { return 7; }
casadi_int fwd1_interp_wh_n_out(void) { return 1; }

const casadi_int* fwd1_interp_wh_sparsity_in(casadi_int i) {
    if (i == 3) return empty_sparsity;  /* out_value not used */
    return scalar_sparsity;
}

const casadi_int* fwd1_interp_wh_sparsity_out(casadi_int i) {
    return scalar_sparsity;
}

int fwd1_interp_wh_work(casadi_int *sz_arg, casadi_int *sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 7;
    if (sz_res) *sz_res = 1;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

int fwd1_interp_wh_alloc_mem(void) { return 0; }
int fwd1_interp_wh_init_mem(int mem) { return 0; }
void fwd1_interp_wh_free_mem(int mem) { }
int fwd1_interp_wh_checkout(void) { return 0; }
void fwd1_interp_wh_release(int mem) { }
void fwd1_interp_wh_incref(void) { }
void fwd1_interp_wh_decref(void) { }

/* ============================================================
 * Reverse mode derivative: adj1_interp_wh
 * Inputs: (ptr, y, theta, out_value, adj_value)
 * Outputs: (adj_ptr, adj_y, adj_theta)
 * ============================================================ */

int adj1_interp_wh(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
    uint64_t hash = (uint64_t)arg[0][0];
    TriInterpData *data = get_or_load(hash);
    if (!data) {
        if (res[0]) res[0][0] = NAN;
        if (res[1]) res[1][0] = NAN;
        if (res[2]) res[2][0] = NAN;
        return 1;
    }
    double y = arg[1][0];
    double theta = arg[2][0];
    /* arg[3] = out_value (not used) */
    double adj = arg[4] ? arg[4][0] : 0;

    double value, dv_dy, dv_dt;
    find_triangle_interp_with_grad(y, theta, data, &value, &dv_dy, &dv_dt);

    if (res[0]) res[0][0] = 0.0;  /* adj w.r.t. hash (constant) */
    if (res[1]) res[1][0] = dv_dy * adj;
    if (res[2]) res[2][0] = dv_dt * adj;
    return 0;
}

casadi_int adj1_interp_wh_n_in(void) { return 5; }
casadi_int adj1_interp_wh_n_out(void) { return 3; }

const casadi_int* adj1_interp_wh_sparsity_in(casadi_int i) {
    if (i == 3) return empty_sparsity;  /* out_value not used */
    return scalar_sparsity;
}

const casadi_int* adj1_interp_wh_sparsity_out(casadi_int i) {
    return scalar_sparsity;
}

int adj1_interp_wh_work(casadi_int *sz_arg, casadi_int *sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
    if (sz_arg) *sz_arg = 5;
    if (sz_res) *sz_res = 3;
    if (sz_iw) *sz_iw = 0;
    if (sz_w) *sz_w = 0;
    return 0;
}

int adj1_interp_wh_alloc_mem(void) { return 0; }
int adj1_interp_wh_init_mem(int mem) { return 0; }
void adj1_interp_wh_free_mem(int mem) { }
int adj1_interp_wh_checkout(void) { return 0; }
void adj1_interp_wh_release(int mem) { }
void adj1_interp_wh_incref(void) { }
void adj1_interp_wh_decref(void) { }
