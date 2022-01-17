#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "nn.h"
#include "nnpi.h"
#include "nncommon.h"
#include "delaunay_internal.h"



int interpolate_from(double *xc, double *yc, double *zc, int nin, double *xout, double *yout, double *zout, int* bad_markers,
	   	int nout)
{

	//printf(" Start! ");

	point* pin = NULL;
	delaunay* d = NULL;
	nnhpi* nn = NULL;
	
	int i;

	//int nout = 0;
	point* pout = NULL;

	pin = malloc(nin * sizeof(point));

	for (i = 0; i < nin; ++i) {
		point* p = &pin[i];

		p->x = xc[i];
		p->y = yc[i];
		p->z = zc[i];
	}

	d = delaunay_build(nin, pin, 0, NULL, 0, NULL);

	//points_generate(2.0, 3.0, 2.0, 3.0, 3, 3, &nout, &pout);
	
	pout = malloc(nout * sizeof(point));

	for (i = 0; i < nout; ++i) {
		point* p = &pout[i];
		p->x = xout[i];
		p->y = yout[i];
	}

	nn = nnhpi_create(d, nout);

	for (i = 0; i < nout; ++i) {
		point* p = &pout[i];
        bad_markers[i] = nnhpi_interpolate_output_marker(nn, p);
		zout[i] = p-> z;
		//printf(" (%f, %f, %f)\n ", p->x, p->y, p->z);
	}

	nnhpi_destroy(nn);
	free(pout);
	delaunay_destroy(d);
	free(pin);
	//printf(" Done! ");

	return 0;
}

int interpolate_weights_from(
		double *xc,
	   	double *yc,
	   	double *zc,
	   	int nin,
	   	double *xout,
	   	double *yout,
	   	int nout,
	   	double *weights_out,
		int *neighbor_index,
		int max_nneighbor)
{

	//printf(" Start! ");

	point* pin = NULL;
	delaunay* d = NULL;
	nnhpi* nn = NULL;
	
	int i;

	//int nout = 0;
	point* pout = NULL;

	pin = malloc(nin * sizeof(point));

	for (i = 0; i < nin; ++i) {
		point* p = &pin[i];

		p->x = xc[i];
		p->y = yc[i];
		p->z = zc[i];
	}

	d = delaunay_build(nin, pin, 0, NULL, 0, NULL);

	//points_generate(2.0, 3.0, 2.0, 3.0, 3, 3, &nout, &pout);
	
	pout = malloc(nout * sizeof(point));

	for (i = 0; i < nout; ++i) {
		point* p = &pout[i];
		p->x = xout[i];
		p->y = yout[i];
	}

	nn = nnhpi_create(d, nout);

	for (i = 0; i < nout; ++i) {
		point* p = &pout[i];
		nnhpi_interpolate_get_weights(nn, p, weights_out, neighbor_index, max_nneighbor, i);
		//printf(" (%f, %f, %f)\n ", p->x, p->y, p->z);
	}


	nnhpi_destroy(nn);
	free(pout);
	delaunay_destroy(d);
	free(pin);
	//printf(" Done! ");

	return 0;
}
