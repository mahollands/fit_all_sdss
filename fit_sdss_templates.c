/**********************************************************/
/* This program fits templates to all SDSS spectra.       */
/* For each SDSS spectrum, firstly the S/N is calculated. */
/* Then for each template, the they are interpolated onto */
/* the same wavelength axis as the SDSS spectrum.         */
/* These are optimally fitted to the SDSS spectrum.       */
/* The template with the minimum chi2 is the winner.      */
/* The loop over files uses all available CPU cores.      */
/* The SDSS name, best template, chisq and S/N are        */
/* finally written to disk.                               */
/*                                                        */
/* Compile with:                                          */
/*                                                        */
/* gcc -o outname dzFinder2.c -O3 -fopenmp -march=native  */
/*                                                        */
/* OR                                                     */
/*                                                        */
/* module load intel/2019.3.199-GCC-8.3.0-2.32            */
/* icc -o outname fit_sdss_templates.c -qopenmp -xhost    */
/*                                                        */
/* Last update 2020-05-02                                 */
/**********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

//#define N_SDSS 5789198
//#define N_SDSS 57892
#define N_SDSS 10000
#define N_PX_MAX 6000
#define N_TEMPLATES 3078

#define X_MIN 3600.
#define X_MAX 9000.
#define SN_CUT 0.

#define DIR_TEMPLATES "../input_data/templates_text"
#define F_TEMPLATE_LIST "../input_data/templates-CV-200430.txt"

#define F_IN        "../input_data/SDSS_dr16_spectra_binary_all.dat"
//#define F_OUT       "../output_data/sdss_dr16_all_CVs_200430.csv"
#define F_OUT       "../output_data/test_output.csv"

/*data structures for SDSS spectra and templates*/
typedef struct{
  unsigned int N;
  char name[32];
  double *x;
  double *y;
}template;

typedef struct{
  unsigned int N;
  double sn;
  double *x;
  double *y;
  double *ivar;
  double *Ti; /*Interpolated template fluxes*/
}spectrum;

/*Function Prototypes*/
void load_templates(template *TT);
void progress_bar(unsigned int n, unsigned int N);
unsigned int set_N_threads(int argc, char **argv);
void parallel_section(template *TT, FILE *input, FILE *output);
void process_pixels(float *data_buffer, unsigned int N_file, spectrum *Sptr);
void interpolate_template(template T, spectrum S);
double calc_chisq(spectrum S);

/******************************* End preamble ********************************/

int main(int argc, char** argv)
{
  template TT[N_TEMPLATES];
  FILE *input, *output;
  double tic, toc;
  unsigned int i;

  /*Count how many threads we're using if user supplied*/
  set_N_threads(argc, argv);

  /*Load templates and list of SDSS file names*/
  load_templates(TT);

  /*ready file I/O file*/
  input = fopen(F_IN, "rb");
  output = fopen(F_OUT, "w");

  /***************************/
  /*loop through SDSS spectra*/
  /***************************/
  puts("starting");
  tic = omp_get_wtime();
  #pragma omp parallel default(shared)
  parallel_section(TT, input, output);
  progress_bar(N_SDSS, N_SDSS);
  toc = omp_get_wtime();
  printf("\ntime to complete = %.3f s\n", toc-tic);

  /*tidy up before program end*/
  for(i=0; i<N_TEMPLATES; ++i)
  {
    free(TT[i].x);
    free(TT[i].y);
  }
  fclose(input);
  fclose(output);

  return 0;
}

/*Loads all templates into an array of templates*/
void load_templates(template *TT)
{
  unsigned int i, j;
  double temp1, temp2;
  FILE *input;
  char tname_full[128];
  template T;

  puts("loading templates");

  /*Read template list and memory allocation in serial*/
  input = fopen(F_TEMPLATE_LIST, "r");
  if(input == NULL)
  {
    printf("Error opening %s\n", F_TEMPLATE_LIST);
    exit(EXIT_FAILURE);
  }
  for(i=0; i<N_TEMPLATES; ++i)
  {
    fscanf(input, "%s %d\n", TT[i].name, &TT[i].N);
    TT[i].x = (double*)malloc(TT[i].N*sizeof(double));
    TT[i].y = (double*)malloc(TT[i].N*sizeof(double));
  }
  fclose(input);

  /*Read template files in parallel*/
  #pragma omp parallel for default(shared) private(i,j,input,tname_full,temp1,temp2,T)
  for(i=0; i<N_TEMPLATES; ++i)
  {
    T = TT[i]; /*shorthand*/

    sprintf(tname_full, "%s/%s", DIR_TEMPLATES, T.name);
    input = fopen(tname_full, "r");
    if(input == NULL)
    {
      printf("Error opening %s\n", tname_full);
      exit(EXIT_FAILURE);
    }

    switch(T.name[0])
    {
      /*two column files*/
      case 'd': /*wd template*/
      case 'D': /*D6 template*/
      case 'G': /*GD492*/
        for(j=0; j<T.N; ++j)
        {
          fscanf(input, "%lf %lf\n", &T.x[j], &T.y[j]);
        }
      break;

      /*four column files*/
      case 'C': /*CVs*/
      case 'h': /*high quality*/
        for(j=0; j<T.N; ++j)
        {
          fscanf(input, "%lf %lf %lf %lf\n", &T.x[j], &T.y[j], &temp1, &temp2);
        }
      break;

      default:
        puts("Problem with template names");
        exit(EXIT_FAILURE);
    }
    fclose(input);
  }
  puts("loaded templates into memory");
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
  for(i=0; i<n1; ++i)
    putchar('*');
  for(i=0; i<n2; ++i)
    putchar('-');
  putchar('|');
  fflush(stdout);
}

/*Use user-specified number of threads if given*/
unsigned int set_N_threads(int argc, char **argv)
{
  unsigned int Nthreads;

  if(argc == 1)
  {
    Nthreads = omp_get_max_threads();
  }
  else
  {
    sscanf(argv[1], "%d", &Nthreads);
    omp_set_num_threads(Nthreads);
  }
  printf("N_threads = %d \n", Nthreads);
  return Nthreads;
}

/*Function that loops over SDSS spectra and runs in parallel*/
void parallel_section(template *TT, FILE *input, FILE *output)
{
  spectrum S;
  unsigned int ispec, itmplt, imin=0;
  unsigned int N_file;
  float data_buffer[3*N_PX_MAX];
  unsigned int intbuffer[4];
  double chisq, chisq_min, chisq_min_red;
  unsigned int plate, mjd, fiber;
  double spec_mem[4*N_PX_MAX];

  /*Asign memory for spectrum*/
  S.x    = spec_mem;
  S.y    = spec_mem +   N_PX_MAX;
  S.ivar = spec_mem + 2*N_PX_MAX;
  S.Ti   = spec_mem + 3*N_PX_MAX;

  /*start parallel for loop. (dynamic,1) is fastest, as each thread works on
  one file at a time, and progress bar usually updates correctly*/
  #pragma omp for schedule(dynamic,1)
  for(ispec=0; ispec<N_SDSS; ++ispec)
  {
    /* READ SDSS SPECTRUM */
    #pragma omp critical
    {
      fread(intbuffer, sizeof(unsigned int), 4, input);
      N_file = intbuffer[0];
      fread(data_buffer, sizeof(float), 3*N_file, input);
    }
    plate = intbuffer[1];
    mjd   = intbuffer[2];
    fiber = intbuffer[3];

    /*process buffered pixel data and calculate S/N; ignore noisy or broken spectra*/
    process_pixels(data_buffer, N_file, &S);
    if(S.sn < SN_CUT || isnan(S.sn) || S.N < 1000) continue; 

    /****************************************/
    /* MAIN CALCULATION LOOP OVER TEMPLATES */
    /****************************************/
    for(chisq_min=DBL_MAX, itmplt=0; itmplt<N_TEMPLATES; ++itmplt)
    {
      interpolate_template(TT[itmplt], S);
      chisq = calc_chisq(S);

      if(chisq < chisq_min)
      {
        chisq_min = chisq; 
        imin = itmplt;
      }
    }

    chisq_min_red = chisq_min / (double)(S.N-1);
    /*output to file*/
    fprintf(output, "%05d-%05d-%04d,%s,%.3e,%.3e\n", plate, mjd, fiber, TT[imin].name, chisq_min_red, S.sn);

    /*progress bar*/
    if(ispec % 100 == 0)
      progress_bar(ispec, N_SDSS);
  }
}

/*Process the SDSS pixel data and store and store in spectrum type*/
void process_pixels(float *data_buffer, unsigned int N_file, spectrum *S)
{
  unsigned int i, N_sn=0, N=0;
  double x, sigma;
  double sn=0;

  /*Loop over sdss spectrum pixels*/
  for(i=0; i<N_file; ++i)
  {
    x = (double)data_buffer[3*i];
    if(x<X_MIN) /*make sure we only load data covered by the templates*/
      continue;
    else if(x>X_MAX)
      break;
    if(x > 5560. && x < 5590.) /*bad sky subtraction line in SDSS*/
      continue;
    S->x[N] = x;
    S->y[N] = (double)data_buffer[3*i+1];
    sigma = (double)data_buffer[3*i+2];
    S->ivar[N] = 1/(sigma*sigma);

    /*Also do signal to noise*/
    if(x > 4500. && x < 6000.)
    {
      sn += S->y[N] / sigma;
      ++N_sn;
    }
    ++N; /*Count useful pixels (N<=N_file)*/
  }
  sn /= (double)N_sn;

  S->sn = sn;
  S->N = N;
}

/*Interpolate a template onto the spectrum axis and store in spectrum*/
void interpolate_template(template T, spectrum S)
{
  unsigned int i, j=1;

  for(i=0; i<S.N; ++i) /*only need to condition the loop on i, since spec stops before template*/
  {
    while(T.x[j] < S.x[i]) ++j;
    S.Ti[i] = T.y[j-1] + (S.x[i]-T.x[j-1])*(T.y[j]-T.y[j-1])/(T.x[j]-T.x[j-1]);
  }
}

/*Calculate the chisq between the template and interpolated spectrum*/
double calc_chisq(spectrum S)
{
  unsigned int i;
  double Sum_st=0, Sum_tt=0, A, temp, chisq=0;

  /* Find optimal scaling parameter*/
  #pragma omp simd private(temp) reduction(+:Sum_tt,Sum_st)
  for(i=0; i<S.N; ++i)
  {
    temp = S.Ti[i] * S.ivar[i];
    Sum_tt += S.Ti[i] * temp;
    Sum_st += S.y[i] * temp;
  }
  /* optimal A */
  A = Sum_st/Sum_tt; 

  /*calc chisq with optimal A*/
  #pragma omp simd private(temp) reduction(+:chisq)
  for(i=0; i<S.N; ++i)
  {
    temp = S.y[i] - A*S.Ti[i];
    chisq += temp * temp * S.ivar[i];
  }
  return chisq;
}
