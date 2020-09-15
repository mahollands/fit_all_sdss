/* This program fits templates to all SDSS spectra. For each SDSS spectrum,
 * firstly the S/N is calculated. Then for each template, the they are
 * interpolated onto the same wavelength axis as the SDSS spectrum. These are
 * optimally fitted to the SDSS spectrum. The template with the minimum chi2
 * is the winner. The loop over files uses all available CPU cores. The SDSS
 * name, best template, chisq and S/N are finally written to disk.          
 *                                                                            
 * Compile with:                                                              
 *                                                                            
 * gcc -o outname fit_sdss_templates.c -O3 -lm -fopenmp -march=native         
 *                                                                            
 * OR                                                                         
 *                                                                            
 * module load intel/2019.3.199-GCC-8.3.0-2.32                                
 * icc -o outname fit_sdss_templates.c -qopenmp -xhost                        
 *                                                                            
 * Last update 2020-09-15
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

//#define N_SDSS 5789198
//#define N_SDSS 578920
#define N_SDSS 10000
#define N_PX_MAX 6000
#define N_TEMPLATES 3986

#define X_MIN 3600.
#define X_MAX 9000.
#define SN_CUT 0.

#define DIR_TEMPLATES "../input_data/templates_text"
#define F_TEMPLATE_LIST "../input_data/grid_all-180809.txt"

#define F_IN        "../input_data/SDSS_dr16_spectra_binary_all.dat"
//#define F_OUT       "../output_data/sdss_dr16_all_CVs_200430.csv"
#define F_OUT       "../output_data/test_output.csv"

/*data structures for SDSS spectra and templates*/
typedef struct {
    unsigned int n;
    char name[32];
    double *x;
    double *y;
}template;

typedef struct {
    unsigned int n;
    double sn;
    double *x;
    double *y;
    double *ivar;
    double *Ti; /*Interpolated template fluxes*/
}spectrum;

struct result{
    unsigned int imin;
    double chisq_min_red;
};

/*Function Prototypes*/
void load_templates(template *Tlist);
void load_templates_names_lengths(template *Tlist);
void load_template_data(template T);
void progress_bar(unsigned int n, unsigned int N);
unsigned int set_n_threads(int argc, char **argv);
void process_sdss_spectra(template *Tlist, FILE *input, FILE *output);
struct result process_sdss_spectrum(template *Tlist, spectrum S);
void process_pixel_data(float *data_buffer, unsigned int n_file, spectrum *Sptr);
void interpolate_template(template T, spectrum S);
double calc_chisq(spectrum S);

/******************************* End preamble ********************************/

int main(int argc, char** argv)
{
    template Tlist[N_TEMPLATES];
    FILE *input, *output;
    double tic, toc;
    unsigned int i;
  
    set_n_threads(argc, argv); /*Set thread count if user supplied*/
    load_templates(Tlist);
  
    /*ready file I/O file*/
    input = fopen(F_IN, "rb");
    output = fopen(F_OUT, "w");
  
    /***************************/
    /*loop through SDSS spectra*/
    /***************************/
    puts("starting");
    tic = omp_get_wtime();
    #pragma omp parallel default(shared)
    process_sdss_spectra(Tlist, input, output);
    progress_bar(N_SDSS, N_SDSS);
    toc = omp_get_wtime();
    printf("\ntime to complete = %.3f s\n", toc-tic);
  
    /*tidy up before program end*/
    for(i=0; i<N_TEMPLATES; i++) {
        free(Tlist[i].x);
        free(Tlist[i].y);
    }
    fclose(input);
    fclose(output);
  
    return EXIT_SUCCESS;
}

/*Loads all templates into an array of templates*/
void load_templates(template *Tlist)
{
    unsigned int i;

    puts("loading templates");
    load_templates_names_lengths(Tlist);
    #pragma omp parallel for default(shared) private(i)
    for(i=0; i<N_TEMPLATES; i++) {
        Tlist[i].x = (double*)malloc(Tlist[i].n*sizeof(double));
        Tlist[i].y = (double*)malloc(Tlist[i].n*sizeof(double));
        load_template_data(Tlist[i]);
    }
    puts("loaded templates into memory");
}

void load_templates_names_lengths(template *Tlist)
{
    unsigned int i;
    FILE *input;

    input = fopen(F_TEMPLATE_LIST, "r");
    if(input == NULL) {
        printf("Error opening %s\n", F_TEMPLATE_LIST);
        exit(EXIT_FAILURE);
    }
    for(i=0; i<N_TEMPLATES; i++)
        fscanf(input, "%s %d\n", Tlist[i].name, &Tlist[i].n);
    fclose(input);
}

void load_template_data(template T)
{
    unsigned int i;
    FILE *input;
    char tname_full[128];

    sprintf(tname_full, "%s/%s", DIR_TEMPLATES, T.name);
    input = fopen(tname_full, "r");
    if(input == NULL) {
        printf("Error opening %s\n", tname_full);
        exit(EXIT_FAILURE);
    }

    /*Read first two columns from template file*/
    for(i=0; i<T.n; i++) {
        fscanf(input, "%lf %lf%*[^\n]\n", &T.x[i], &T.y[i]);
    }
    fclose(input);
}

/*Prints a progress bar to screen*/
void progress_bar(unsigned int n, unsigned int N)
{
    unsigned int Nbar = 30, n1, n2, i;
    float frac;

    frac = (float)n/(float)N;
    n1 = (int)(frac*(float)Nbar);
    n2 = Nbar-n1;

    printf("\rspectra processed: %7d/%d ", n, N);
    putchar('|');
    for(i=0; i<n1; i++)
        putchar('*');
    for(i=0; i<n2; i++)
        putchar('-');
    putchar('|');
    fflush(stdout);
}

/*Use user-specified number of threads if given*/
unsigned int set_n_threads(int argc, char **argv)
{
    unsigned int Nthreads;

    if(argc == 1) {
        Nthreads = omp_get_max_threads();
    } else {
        sscanf(argv[1], "%d", &Nthreads);
        omp_set_num_threads(Nthreads);
    }
    printf("N_threads = %d \n", Nthreads);
    return Nthreads;
}

/*Loop over SDSS spectra and running in parallel*/
void process_sdss_spectra(template *Tlist, FILE *input, FILE *output)
{
    unsigned int ispec;
    unsigned int intbuffer[4], *n_file, *plate, *mjd, *fiber;
    float data_buffer[3*N_PX_MAX];
    spectrum S;
    double spec_mem[4*N_PX_MAX];
    struct result r;

    /*Assign memory for spectrum*/
    S.x    = spec_mem;
    S.y    = spec_mem +   N_PX_MAX;
    S.ivar = spec_mem + 2*N_PX_MAX;
    S.Ti   = spec_mem + 3*N_PX_MAX;
    n_file = intbuffer;
    plate = intbuffer + 1;
    mjd   = intbuffer + 2;
    fiber = intbuffer + 3;

    /*start parallel for loop. (dynamic,1) is fastest, as each thread works on
    one file at a time, and progress bar usually updates correctly*/
    #pragma omp for schedule(dynamic,1)
    for(ispec=0; ispec<N_SDSS; ispec++) {
        /* READ SDSS SPECTRUM */
        #pragma omp critical
        {
            fread(intbuffer, sizeof(unsigned int), 4, input);
            fread(data_buffer, sizeof(float), 3*(*n_file), input);
        }
        process_pixel_data(data_buffer, (*n_file), &S);
        if(S.sn < SN_CUT || isnan(S.sn) || S.n < 1000)
            continue; 

        /* Loop over templates */
        r = process_sdss_spectrum(Tlist, S);

        /*output to file*/
        fprintf(output, "%05d-%05d-%04d,%s,%.3e,%.3e\n",
        (*plate), (*mjd), (*fiber),
        Tlist[r.imin].name, r.chisq_min_red, S.sn);

        /*progress bar*/
        if(ispec % 100 == 0)
            progress_bar(ispec, N_SDSS);
    }
}

/*Loop over templates for a single spectrum, finding lowest chisq*/
struct result process_sdss_spectrum(template *Tlist, spectrum S)
{
    unsigned int i;
    double chisq, chisq_min;
    struct result r;

    for(chisq_min=DBL_MAX, i=0; i<N_TEMPLATES; i++) {
        interpolate_template(Tlist[i], S);
        chisq = calc_chisq(S);

        if(chisq < chisq_min) {
            chisq_min = chisq; 
            r.imin = i;
        }
    }
    r.chisq_min_red = chisq_min / (double)(S.n-1);
    return r;
}

/*Process the SDSS pixel data and store and store in spectrum type*/
void process_pixel_data(float *data_buffer, unsigned int n_file, spectrum *S)
{
    unsigned int i, n_sn=0, n=0;
    double x, sigma;
    double sn=0;

    /*Loop over sdss spectrum pixels*/
    for(i=0; i<n_file; i++) {
        x = (double)data_buffer[3*i];
        if(x>X_MAX) /*make sure we only load data covered by the templates*/
            break;
        if(x<X_MIN)
            continue;
        if(x > 5560. && x < 5590.) /*bad sky subtraction line in SDSS*/
            continue;
        S->x[n] = x;
        S->y[n] = (double)data_buffer[3*i+1];
        sigma = (double)data_buffer[3*i+2];
        S->ivar[n] = 1/(sigma*sigma);

        /*Also do signal to noise*/
        if(x > 4500. && x < 6000.) {
            sn += S->y[n] / sigma;
            n_sn++;
        }
        n++; /*Count useful pixels (N<=N_file)*/
    }
    sn /= (double)n_sn;

    S->sn = sn;
    S->n = n;
}

/*Interpolate a template onto the spectrum axis and store in spectrum*/
void interpolate_template(template T, spectrum S)
{
  unsigned int i, j=1;

  for(i=0; i<S.n; i++) { /*only need to condition the loop on i, since spec stops before template*/
      while(T.x[j] < S.x[i])
          j++;
      S.Ti[i] = T.y[j-1] + (S.x[i]-T.x[j-1])*(T.y[j]-T.y[j-1])/(T.x[j]-T.x[j-1]);
  }
}

/*Calculate the chisq between the template and interpolated spectrum*/
double calc_chisq(spectrum S)
{
    unsigned int i;
    double Sum_st=0, Sum_tt=0, A, tmp, chisq=0;

    /* Find optimal scaling parameter*/
    #pragma omp simd private(tmp) reduction(+:Sum_tt,Sum_st)
    for(i=0; i<S.n; i++) {
        tmp = S.Ti[i] * S.ivar[i];
        Sum_tt += S.Ti[i] * tmp;
        Sum_st += S.y[i] * tmp;
    }
    /* optimal A */
    A = Sum_st/Sum_tt; 

    /*calc chisq with optimal A*/
    #pragma omp simd private(tmp) reduction(+:chisq)
    for(i=0; i<S.n; i++) {
        tmp = S.y[i] - A*S.Ti[i];
        chisq += tmp * tmp * S.ivar[i];
    }
    return chisq;
}
