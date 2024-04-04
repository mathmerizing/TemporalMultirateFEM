/* ---------------------------------------------------------------------
 * The space time finite elements code has been based on
 * Step-3 and Step-46 of the deal.II tutorial programs.
 *
 * STEP-3:
 * =======
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 * 
 * ---------------------------------------------------------------------
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 * 
 * ---------------------------------------------------------------------
 * 
 * STEP-46:
 * =======
 *
 * Copyright (C) 2011 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2011
 *
 *
 * ---------------------------------------------------------------------
 * TENSOR-PRODUCT SPACE-TIME FINITE ELEMENTS:
 * ==========================================
 * Tensor-product space-time code for the coupled heat and wave equation with Q^s finite elements in space and dG-Q^r finite elements in time: cG(s)cG(r)
 * For the spatial finite elements, we use FENothing which extend the fluid solution by 0 to the solid domain and vice versa.
 * For a detailed explanation of FENothing check out step-46 from the deal.II tutorials (https://www.dealii.org/current/doxygen/deal.II/step_46.html).
 * We use multirate finite elements in time, where e.g. for each temporal element of the fluid we have four temporal elements of the solid.
 * 
 * Author: Julian Roth, 2022-2023
 * 
 * For more information on tensor-product space-time finite elements please also check out the DTM-project by Uwe Köcher and contributors.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <set>
#include <memory>
#include <sys/stat.h> // for mkdir

// compute conditon numbers of system matrices
#ifndef CONDITION
#  define CONDITION false
#endif

#ifndef DIM
#  define DIM 1
#endif

// use soldi source function
#ifndef SOLID_SOURCE
#  define SOLID_SOURCE true
#endif

using namespace dealii;

double nu = 0.001;
double lambda = 1000.;
#if DIM == 1
	double delta = 0.; // no solid damping in 1+1D
#elif DIM == 2
	double delta = 0.1;
#endif


void print_as_numpy_arrays_high_resolution(SparseMatrix<double> &matrix,
					    std::ostream &     out,
                                            const unsigned int precision)
{
  AssertThrow(out.fail() == false, ExcIO());

  out.precision(precision);
  out.setf(std::ios::scientific, std::ios::floatfield);

  std::vector<int> rows;
  std::vector<int> columns;
  std::vector<double> values;
  rows.reserve(matrix.n_nonzero_elements());
  columns.reserve(matrix.n_nonzero_elements());
  values.reserve(matrix.n_nonzero_elements());

  SparseMatrixIterators::Iterator< double, false > it = matrix.begin();
  for (unsigned int i = 0; i < matrix.m(); i++) {
 	 for (it = matrix.begin(i); it != matrix.end(i); ++it) {
 		rows.push_back(i);
 		columns.push_back(it->column());
		values.push_back(matrix.el(i,it->column()));
 	 }
  }

  for (auto d : values)
    out << d << ' ';
  out << '\n';

  for (auto r : rows)
    out << r << ' ';
  out << '\n';

  for (auto c : columns)
    out << c << ' ';
  out << '\n';
  out << std::flush;

  AssertThrow(out.fail() == false, ExcIO());
}

template<int dim>
class InitialValues: public Function<dim> {
public:
	InitialValues() :
		Function<dim>(2+2) {
	}

	virtual double value(const Point<dim> &p,
			const unsigned int component) const override;

	virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;
};

template<int dim>
double InitialValues<dim>::value(const Point<dim> &/*p*/,
		const unsigned int /*component*/) const {
	return 0.;
}

template <int dim>
void InitialValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<2+2; ++c)
    values(c) = InitialValues<dim>::value(p, c);
}

template<int dim>
class BoundaryValues: public Function<dim> {
public:
	BoundaryValues() : 
		Function<dim>(2+2) {
	}

	virtual double value(const Point<dim> &p,
			const unsigned int component = 0) const override;

	virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;
};

template<int dim>
double BoundaryValues<dim>::value(const Point<dim> &/*p*/,
		const unsigned int /*component*/) const {
	//const double t = this->get_time();
	return 0.;
}

template <int dim>
void BoundaryValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<2+2; ++c)
    values(c) = BoundaryValues<dim>::value(p, c);
}

template <int dim>
class Solution: public Function<dim> {
public:
    Solution() : Function<dim>(2+2) {}

	virtual	double value(const Point<dim> &p,
	        const unsigned int component) const override;
		
	virtual void vector_value (const Point<dim> &p,
	        Vector<double>   &value) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	const double t = this->get_time();

	if (dim == 1)
    {
		if (component == 0)      /* u_f */
			return (1.0/2.0)*std::pow(t, 2)*p[0];
		else if (component == 1) /* v_f */
			return 2*t*std::sin((1.0/4.0)*M_PI*p[0]);
		else if (component == 2) /* u_s */
			return std::pow(t, 2)*std::cos(M_PI*((1.0/2.0)*p[0] - 1));
		else if (component == 3) /* v_s */
			return 2*t*std::cos(M_PI*((1.0/2.0)*p[0] - 1));
	}
	
	return 0.; // no analytical solution is known
}

template <int dim>
void Solution<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const
{
  for (unsigned int c=0; c<2+2; ++c)
    values(c) = Solution<dim>::value(p, c);
}

template<int dim>
class RightHandSide: public Function<dim> {
public:
	RightHandSide() :
			Function<dim>() {
	}
	virtual double value(const Point<dim> &p,
			const unsigned int component) const;
};

template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	const double t = this->get_time();
	
	if (dim == 1)
	{
		if (component == 0)
			return 0.; /* u_f */
		else if (component == 1)
			return (1.0/8.0)*std::pow(M_PI, 2)*t*nu*std::sin((1.0/4.0)*M_PI*p[0]) + 2*std::sin((1.0/4.0)*M_PI*p[0]); /* v_f */  // 1.; // 
		else if (component == 2)
			return 0.; /* u_s */
		else if (component == 3)
			return  (1.0/4.0)*std::pow(M_PI, 2)*std::pow(t, 2)*lambda*std::cos(M_PI*((1.0/2.0)*p[0] - 1)) + 2*std::cos(M_PI*((1.0/2.0)*p[0] - 1)); /* v_s */  // 1.; // 
	}
	else if (dim == 2)
	{
		double temporal_scaling = 1.;
		int t_floor = (t + 1.0e-6);
		if (t - t_floor >= 0.1 + 1e-6)
			temporal_scaling = 0.;

		if (!SOLID_SOURCE)
		{
			// fluid source term
			if (component == 0)
				return 0.; /* u_f */
			else if (component == 1)
				return temporal_scaling * std::exp(-(std::pow(p[0]-0.5, 2) + std::pow(p[1]-0.5, 2))); /* v_f */
			else if (component == 2)
				return 0.;  /* u_s */
			else if (component == 3)
				return 0.;  /* v_s */
		}
		else
		{
			// solid source term
			if (component == 0)
				return 0.; /* u_f */
			else if (component == 1)
				return 0.; /* v_f */
			else if (component == 2)
				return 0.;  /* u_s */
			else if (component == 3)
				return temporal_scaling * std::exp(-(std::pow(p[0]-0.5, 2) + std::pow(p[1]+0.5, 2)));  /* v_s */
		}
	}

	return 0.;
}

class Slab {
public:
	// constructor
	Slab(unsigned int r, double start_time, double end_time);

	//////////
	// fluid
	//
	// time
	Triangulation<1> fluid_time_triangulation;
	FE_DGQArbitraryNodes<1> fluid_time_fe;
	DoFHandler<1> fluid_time_dof_handler;

	//////////
	// solid
	//
	// time
	Triangulation<1> solid_time_triangulation;
	FE_DGQArbitraryNodes<1> solid_time_fe;
	DoFHandler<1> solid_time_dof_handler;

	double start_time, end_time;
};

// NOTE: Use QGaussLobatto quadrature in time. For QGauss one would need to change how the initial value is computed for the next slab in run().
//       Moreover, the temporal interpolation matrix would need to be adapted.
Slab::Slab(unsigned int r, double start_time, double end_time) :
		fluid_time_fe(QGaussLobatto<1>(r+1)), fluid_time_dof_handler(fluid_time_triangulation), solid_time_fe(
				QGaussLobatto<1>(r+1)), solid_time_dof_handler(solid_time_triangulation), start_time(
				start_time), end_time(end_time) {
}

template<int dim>
class SpaceTime {
public:
	SpaceTime(int s, 
			std::vector<unsigned int> r,
			std::vector<double> time_points = { 0., 1. },
			unsigned int max_n_refinement_cycles = 3,
			unsigned int initial_temporal_ref_fluid = 0,
			unsigned int initial_temporal_ref_solid = 1,
			bool refine_space = true,
			bool refine_time = true,
			bool split_slabs = true,
			double gamma = 0.);
	void run();
	void print_grids(std::string file_name_space, std::string file_name_time_fluid, std::string file_name_time_solid, std::string file_name_time_joint);
	void print_convergence_table();

private:
	void make_grids();
	void set_active_fe_indices();
	void setup_system(std::shared_ptr<Slab> &slab, unsigned int k);
	void assemble_system(std::shared_ptr<Slab> &slab, bool assemble_matrix);
	void apply_boundary_conditions(std::shared_ptr<Slab> &slab);
	void solve(bool invert);
	void get_solution_on_finer_mesh(std::shared_ptr<Slab> &slab, std::vector<Vector<double>> &solution_at_t_qq, std::vector<double> &fluid_values_t_qq, std::vector<double> &solid_values_t_qq);
	void output_results(std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number, bool last_slab);
	void print_coordinates(std::shared_ptr<Slab> &slab, std::string output_dir, unsigned int slab_number);
	void print_solution(std::shared_ptr<Slab> &slab, std::string output_dir, unsigned int slab_number);
	void print_error(std::shared_ptr<Slab> &slab, std::string output_dir, unsigned int slab_number);
	void process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab);
	void compute_goal_functional(std::shared_ptr<Slab> &slab);

	enum
	{
		fluid_domain_id,
		solid_domain_id
	};

	//////////
	// space
	//
	Triangulation<dim>    space_triangulation;
	FESystem<dim>         fluid_space_fe;
	FESystem<dim>         solid_space_fe;
  	hp::FECollection<dim> space_fe_collection;
  	DoFHandler<dim>       space_dof_handler;
	Vector<double> 		  initial_solution_fluid;
	Vector<double> 		  initial_solution_solid;
	
	//////////
	// time
	//
	std::vector<std::shared_ptr<Slab> > slabs;

	// fluid
	std::set< std::pair<double, unsigned int> > fluid_time_support_points; // (time_support_point, support_point_index)

	types::global_dof_index       fluid_n_space_u;
	types::global_dof_index       fluid_n_space_v;
	types::global_dof_index       fluid_n_space_dofs;
	types::global_dof_index       fluid_n_dofs; // space-time DoFs
		
	// solid
	std::set< std::pair<double, unsigned int> > solid_time_support_points; // (time_support_point, support_point_index)

	types::global_dof_index       solid_n_space_u;
	types::global_dof_index       solid_n_space_v;
	types::global_dof_index       solid_n_space_dofs;
	types::global_dof_index       solid_n_dofs; // space-time DoFs

	// temporal interpolation matrix
	SparsityPattern temporal_interpolation_sparsity_pattern;
	SparseMatrix<double> temporal_interpolation_matrix;

	///////////////////	
	// space-time
	//
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> system_matrix;
	SparseDirectUMFPACK A_direct;
	Vector<double> solution;
	Vector<double> system_rhs;

	const FEValuesExtractors::Scalar fluid_displacement = 0;
  	const FEValuesExtractors::Scalar fluid_velocity     = 1;
	const FEValuesExtractors::Scalar solid_displacement = 2;
  	const FEValuesExtractors::Scalar solid_velocity     = 3;

	double start_time, end_time;
	
	unsigned int n_snapshots;
	unsigned int max_n_refinement_cycles;
	unsigned int initial_temporal_ref_fluid;
	unsigned int initial_temporal_ref_solid;
	bool refine_space, refine_time, split_slabs;

	double gamma = 0.; // penalty parameter

	double L2_fluid_error, L2_solid_error;
	double L2_error;
	double goal_func_value;
	std::vector<double> goal_func_vals;
	std::vector<double> L2_error_vals;
	std::vector<double> L2_fluid_error_vals;
	std::vector<double> L2_solid_error_vals;
	unsigned int fluid_total_n_dofs;
	unsigned int solid_total_n_dofs;
	ConvergenceTable convergence_table;
};

template <int dim>
SpaceTime<dim>::SpaceTime(
	int s, 
	std::vector<unsigned int> r, 
	std::vector<double> time_points,
	unsigned int max_n_refinement_cycles,
	unsigned int initial_temporal_ref_fluid,
	unsigned int initial_temporal_ref_solid,
	bool refine_space,
	bool refine_time,
	bool split_slabs,
	double gamma) : 
		fluid_space_fe(
			/*u_f*/ FE_Q<dim>(s), 1,
			/*v_f*/ FE_Q<dim>(s), 1,
			/*u_s*/ FE_Nothing<dim>(), 1,
			/*v_s*/ FE_Nothing<dim>(), 1),
		solid_space_fe(
			/*u_f*/ FE_Nothing<dim>(), 1,
			/*v_f*/ FE_Nothing<dim>(), 1,
			/*u_s*/ FE_Q<dim>(s), 1,
			/*v_s*/ FE_Q<dim>(s), 1),
		space_dof_handler(space_triangulation),

		max_n_refinement_cycles(max_n_refinement_cycles),
		initial_temporal_ref_fluid(initial_temporal_ref_fluid),
		initial_temporal_ref_solid(initial_temporal_ref_solid),
		refine_space(refine_space),
		refine_time(refine_time),
		split_slabs(split_slabs),
		gamma(gamma)
{
	// time_points = [t_0, t_1, ..., t_M]
	// r = [r_1, r_2, ..., r_M] with r_k is the temporal FE degree on I_k = (t_{k-1},t_k]
	Assert(r.size() + 1 == time_points.size(),
		   ExcDimensionMismatch(r.size() + 1, time_points.size()));
	// NOTE: at first hard coding r = 1 as a temporal degree
	for (unsigned int k = 0; k < r.size(); ++k)
		Assert(r[k] == 1, ExcNotImplemented());

	// create slabs
	for (unsigned int k = 0; k < r.size(); ++k)
		slabs.push_back(
			std::make_shared<Slab>(r[k], time_points[k],
								   time_points[k + 1]));

	start_time = time_points[0];
	end_time = time_points[time_points.size() - 1];

	space_fe_collection.push_back(fluid_space_fe);
	space_fe_collection.push_back(solid_space_fe);
}

void print_1d_grid(std::ofstream &out, Triangulation<1> &triangulation, double start, double end) {
	out << "<svg width='1200' height='200' xmlns='http://www.w3.org/2000/svg' version='1.1'>" << std::endl;
	out << "<rect fill='white' width='1200' height='200'/>" << std::endl;
	out << "  <line x1='100' y1='100' x2='1100' y2='100' style='stroke:black;stroke-width:4'/>" << std::endl; // timeline
	out << "  <line x1='100' y1='125' x2='100' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl; // first tick

	// ticks
	for (auto &cell : triangulation.active_cell_iterators())
		out << "  <line x1='" << (int)(1000*(cell->vertex(1)[0]-start)/(end-start)) + 100 <<"' y1='125' x2='" 
		    << (int)(1000*(cell->vertex(1)[0]-start)/(end-start)) + 100 <<"' y2='75' style='stroke:black;stroke-width:4'/>" << std::endl;

	out << "</svg>" << std::endl;
}

void print_1d_grid_slabwise(std::ofstream &out,
		std::vector<std::shared_ptr<Slab> > &slabs, double start, double end,
		bool fluid) {
	out << "<svg width='1200' height='200' xmlns='http://www.w3.org/2000/svg' version='1.1'>"
		<< std::endl;
	out << "<rect fill='white' width='1200' height='200'/>" << std::endl;

	for (unsigned int k = 0; k < slabs.size(); ++k)
		out << "  <line x1='"
			<< (int) (1000 * (slabs[k]->start_time - start) / (end - start)) + 100
			<< "' y1='100' x2='"
			<< (int) (1000 * (slabs[k]->end_time - start) / (end - start)) + 100
			<< "' y2='100' style='stroke:"
			<< ((k % 2) ? "blue" : "black") << ";stroke-width:4'/>"
			<< std::endl; // timeline

	out << "  <line x1='100' y1='125' x2='100' y2='75' style='stroke:black;stroke-width:4'/>"
		<< std::endl; // first tick

	// ticks
	for (auto &slab : slabs)
	{
		auto time_triangulation = ((fluid) ?  &(slab->fluid_time_triangulation) : &(slab->solid_time_triangulation));
		for (auto &cell : time_triangulation->active_cell_iterators())
			out << "  <line x1='"
				<< (int) (1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100
				<< "' y1='125' x2='"
				<< (int) (1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100
				<< "' y2='75' style='stroke:black;stroke-width:4'/>"
				<< std::endl;
	}

	out << "</svg>" << std::endl;
}

void print_1d_grid_slabwise_joint(std::ofstream &out,
		std::vector<std::shared_ptr<Slab> > &slabs, double start, double end) {
	// print joint fluid/solid temporal triangulation, e.g. fluid above timeline and solid below timeline
	out << "<svg width='1200' height='200' xmlns='http://www.w3.org/2000/svg' version='1.1'>"
		<< std::endl;
	out << "<rect fill='white' width='1200' height='200'/>" << std::endl;

	for (unsigned int k = 0; k < slabs.size(); ++k)
		out << "  <line x1='"
			<< (int) (1000 * (slabs[k]->start_time - start) / (end - start)) + 100
			<< "' y1='100' x2='"
			<< (int) (1000 * (slabs[k]->end_time - start) / (end - start)) + 100
			<< "' y2='100' style='stroke:"
			<< ((k % 2) ? "blue" : "black") << ";stroke-width:4'/>"
			<< std::endl; // timeline

	out << "  <line x1='100' y1='125' x2='100' y2='75' style='stroke:black;stroke-width:4'/>"
		<< std::endl; // first tick

	// ticks
	for (auto &slab : slabs)
	{
		int offset = 0;
		for (auto time_triangulation : {&(slab->solid_time_triangulation), &(slab->fluid_time_triangulation)})
		{
			for (auto &cell : time_triangulation->active_cell_iterators())
				out << "  <line x1='"
					<< (int) (1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100
					<< "' y1='" << 100 + offset*25 << "' x2='"
					<< (int) (1000 * (cell->vertex(1)[0] - start) / (end - start)) + 100
					<< "' y2='" << 75 + offset*25 <<"' style='stroke:black;stroke-width:4'/>"
					<< std::endl;
			offset += 1;
		}
	}

	out << "</svg>" << std::endl;
}

template<>
void SpaceTime<2>::print_grids(std::string file_name_space, std::string file_name_time_fluid, std::string file_name_time_solid, std::string file_name_time_joint) {

	//////////
	// space
	//
	
	std::ofstream out_space(file_name_space);
	GridOut grid_out_space;
	grid_out_space.write_svg(space_triangulation, out_space);

	//////////
	// time
	//

	// fluid
	std::ofstream out_time_fluid(file_name_time_fluid);
	print_1d_grid_slabwise(out_time_fluid, slabs, start_time, end_time, true);

	// solid
	std::ofstream out_time_solid(file_name_time_solid);
	print_1d_grid_slabwise(out_time_solid, slabs, start_time, end_time, false);

	// joint: fluid + solid
	std::ofstream out_time_joint(file_name_time_joint);
	print_1d_grid_slabwise_joint(out_time_joint, slabs, start_time, end_time);
}

template<>
void SpaceTime<1>::print_grids(std::string file_name_space, std::string file_name_time_fluid, std::string file_name_time_solid, std::string file_name_time_joint) {

	//////////
	// space
	//
		
	std::ofstream out_space(file_name_space);
	print_1d_grid(out_space, space_triangulation, 0., 4.);
		
	/////////
	// time
    //

    // fluid
	std::ofstream out_time_fluid(file_name_time_fluid);
	print_1d_grid_slabwise(out_time_fluid, slabs, start_time, end_time, true);

	// solid
	std::ofstream out_time_solid(file_name_time_solid);
	print_1d_grid_slabwise(out_time_solid, slabs, start_time, end_time, false);

	// joint: fluid + solid
	std::ofstream out_time_joint(file_name_time_joint);
	print_1d_grid_slabwise_joint(out_time_joint, slabs, start_time, end_time);
}

template<>
void SpaceTime<1>::make_grids() {
	//////////
	// space
	//
	
	// Ω_f = (0,2), Ω_s = (2,4)
	GridGenerator::hyper_rectangle(space_triangulation, Point<1>(0.), Point<1>(4.));
	space_triangulation.refine_global(1); // need at least one spatial cell each for fluid and solid

	// mark boundaries: x=0: homogeneous Dirichlet (fluid), x=4: homogeneous Neumann (solid)
	for (auto &cell : space_triangulation.cell_iterators())
	    for (unsigned int face = 0; face < GeometryInfo<1>::faces_per_cell;face++)
		    if (cell->face(face)->at_boundary())
				if (cell->face(face)->center()[0] > 3.9)
					cell->face(face)->set_boundary_id(2); // Neumann boundary
				else if (cell->face(face)->center()[0] < 0.1)
					cell->face(face)->set_boundary_id(0); // Dirichlet boundary

    // // checking correctness of boundary ids
	// for (auto &cell : space_triangulation.active_cell_iterators())
	//     for (unsigned int face = 0; face < GeometryInfo<1>::faces_per_cell;face++)
	// 	    if (cell->face(face)->at_boundary()) 
	// 			std::cout << "x = " << cell->face(face)->center()[0] << ", id = " << cell->face(face)->boundary_id() << std::endl;

	// mark fluid and solid domain by setting the material_id
	for (const auto &cell : space_dof_handler.active_cell_iterators())
		if (cell->center()[0] < 2.)
			cell->set_material_id(fluid_domain_id);
		else
			cell->set_material_id(solid_domain_id);

	// globally refine the spatial grid
	space_triangulation.refine_global(2);

	// // checking correctness of material_id for fluid and solid domain
	// for (const auto &cell : space_dof_handler.active_cell_iterators())
	// 	std::cout << "x = " << cell->center()[0] << ", id = " << cell->material_id() << std::endl;

	//////////
	// time
    //

    // fluid
	for (auto &slab: slabs)
	{
		// create temporal fluid grid on slab
	    GridGenerator::hyper_rectangle(
			slab->fluid_time_triangulation,
			Point<1>(slab->start_time),
			Point<1>(slab->end_time)
		);
		// globally refine the temporal grid
		slab->fluid_time_triangulation.refine_global(initial_temporal_ref_fluid);
	}

	// solid
	for (auto &slab: slabs)
	{
		// create temporal solid grid on slab
	    GridGenerator::hyper_rectangle(
			slab->solid_time_triangulation,
			Point<1>(slab->start_time),
			Point<1>(slab->end_time)
		);
		// globally refine the temporal grid
		slab->solid_time_triangulation.refine_global(initial_temporal_ref_solid);
	}
}

template<>
void SpaceTime<2>::make_grids() {
	//////////
	// space
	//
	
	// Ω_f = (0,4) x (0,1), Ω_s = (0,4) x (-1,0)
	GridGenerator::subdivided_hyper_rectangle(space_triangulation, {/*2,2*/ /*8,2*/ 80,20}, Point<2>(0.,-1.), Point<2>(4.,1.));
	
	// Dirichlet: id == 0, Interface: id == 1, Neumann: id == 2
	// TODO: double check with Martyna whether this is correct!!!
	for (auto &cell : space_triangulation.cell_iterators())
	    for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell;face++)
		    if (cell->face(face)->at_boundary())
				if ((cell->face(face)->center()[0] < 0. + 1e-10 || cell->face(face)->center()[0] > 4. - 1e-10) && cell->center()[1] > 0. + 1e-10)
					cell->face(face)->set_boundary_id(2); // Neumann boundary (fluid)
				else if (cell->face(face)->center()[1] > 1. - 1e-10)
					cell->face(face)->set_boundary_id(0); // Dirichlet boundary (fluid)
				else if (cell->face(face)->center()[1] < -1. + 1e-10)
					cell->face(face)->set_boundary_id(2); // Neumann boundary (solid)
				else if ((cell->face(face)->center()[0] < 0. + 1e-10 || cell->face(face)->center()[0] > 4. - 1e-10) && cell->center()[1] <= 0. + 1e-10)
					cell->face(face)->set_boundary_id(0); // Dirichlet boundary (solid)

    // // checking correctness of boundary ids
	// std::cout << "boundary ids:" << std::endl;
	// for (auto &cell : space_triangulation.active_cell_iterators())
	//     for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell;face++)
	// 	    if (cell->face(face)->at_boundary()) 
	// 			std::cout << "x = " << cell->face(face)->center()[0] << ", y = " << cell->face(face)->center()[1] << ", id = " << cell->face(face)->boundary_id() << std::endl;

	// mark fluid and solid domain by setting the material_id
	for (const auto &cell : space_dof_handler.active_cell_iterators())
		if (cell->center()[1] > 0.)
			cell->set_material_id(fluid_domain_id);
		else
			cell->set_material_id(solid_domain_id);

	// // checking correctness of material_id for fluid and solid domain
	// std::cout << "\nmaterial_ids:" << std::endl;
	// for (const auto &cell : space_dof_handler.active_cell_iterators())
	// 	std::cout << "x = " << cell->center()[0] << ", y = " << cell->center()[1] << ", id = " << cell->material_id() << " (" << ((cell->material_id() == fluid_domain_id) ? "fluid)" : "solid)") << std::endl;

	//////////
	// time
    //

    // fluid
	for (auto &slab: slabs)
	{
		// create temporal fluid grid on slab
	    GridGenerator::hyper_rectangle(
			slab->fluid_time_triangulation,
			Point<1>(slab->start_time),
			Point<1>(slab->end_time)
		);
		// globally refine the temporal grid
		slab->fluid_time_triangulation.refine_global(initial_temporal_ref_fluid);
	}

	// solid
	for (auto &slab: slabs)
	{
		// create temporal solid grid on slab
	    GridGenerator::hyper_rectangle(
			slab->solid_time_triangulation,
			Point<1>(slab->start_time),
			Point<1>(slab->end_time)
		);
		// globally refine the temporal grid
		slab->solid_time_triangulation.refine_global(initial_temporal_ref_solid);
	}
}

template<int dim>
void SpaceTime<dim>::set_active_fe_indices()
{
	for (const auto &cell : space_dof_handler.active_cell_iterators())
	{
		if (cell->material_id() == fluid_domain_id)
			cell->set_active_fe_index(0);
		else if (cell->material_id() == solid_domain_id)
			cell->set_active_fe_index(1);
		else
			Assert(false, ExcNotImplemented());
	}
}

template<int dim>
void SpaceTime<dim>::setup_system(std::shared_ptr<Slab> &slab, unsigned int k) {
	///////////////////
	// distribute DoFs

	// time	
	slab->fluid_time_dof_handler.distribute_dofs(slab->fluid_time_fe);
	slab->solid_time_dof_handler.distribute_dofs(slab->solid_time_fe);

	// number of space-time DoFs per domain
	fluid_n_dofs = fluid_n_space_dofs * slab->fluid_time_dof_handler.n_dofs();
	solid_n_dofs = solid_n_space_dofs * slab->solid_time_dof_handler.n_dofs();

	std::cout << "Slab Q_" << k << " = Ω x (" << slab->start_time << "," << slab->end_time << "):" << std::endl;
	std::cout << "=======================" << std::endl;
	std::cout << "#DoFs:" << std::endl << "------" << std::endl;
	std::cout << "Fluid: " <<  fluid_n_space_dofs
	          << " (" << fluid_n_space_u << '+' << fluid_n_space_v <<  ')' << " (space), "
		  	  << slab->fluid_time_dof_handler.n_dofs() << " (time)" << std::endl;
  	std::cout << "Solid: " <<  solid_n_space_dofs
	          << " (" << solid_n_space_u << '+' << solid_n_space_v <<  ')' << " (space), "
		 	  << slab->solid_time_dof_handler.n_dofs() << " (time)" << std::endl;
	std::cout << "Total: " <<  fluid_n_dofs + solid_n_dofs 
	          << " (space-time)" << std::endl  << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////
	// space-time sparsity pattern = tensor product of spatial and temporal sparsity pattern
	//
	
	// linear problem with constant temporal and spatial meshes: sparsity pattern remains the same on future slabs 
	if (k == 1) {
		// NOTE: For simplicity, we assume that either the fluid or the solid temporal triangulation contains only one element
		Assert(initial_temporal_ref_fluid == 0 || initial_temporal_ref_solid == 0, ExcNotImplemented());
		// NOTE: This one temporal element then has 2 temporal DoFs for dG(1)
		Assert(slab->fluid_time_dof_handler.n_dofs() == 2 || slab->solid_time_dof_handler.n_dofs() == 2,
			ExcNotImplemented());

		/////////////////////////////////
		// temporal interpolation matrix
		double larger_time_n_dofs = std::max(
			slab->fluid_time_dof_handler.n_dofs(),
			slab->solid_time_dof_handler.n_dofs()
		);
		double larger_time_n_cells = std::max(
			slab->fluid_time_triangulation.n_active_cells(),
			slab->solid_time_triangulation.n_active_cells()
		);
		// NOTE: assuming that fluid or solid has only 1 temporal element with 2 DoFs (dG(1))
		DynamicSparsityPattern temp_interp_dsp(larger_time_n_dofs, 2);

		// create sparsity pattern for temporal interpolation matrix by hand
		for (unsigned int i = 0; i < larger_time_n_dofs - 1; ++i) {
			temp_interp_dsp.add(i, 0);
			temp_interp_dsp.add(i + 1, 1);
		}

		temporal_interpolation_sparsity_pattern.copy_from(temp_interp_dsp);
		temporal_interpolation_matrix.reinit(temporal_interpolation_sparsity_pattern);

		// fill temporal_interpolation_matrix
		for (unsigned int i = 0, j = 0; i < larger_time_n_cells; ++i, ++j)
		{
			temporal_interpolation_matrix.add(
				j,
				0,
				(larger_time_n_cells - i) / larger_time_n_cells
			);
			temporal_interpolation_matrix.add(
				(larger_time_n_dofs-1) - j,
				1,
				(larger_time_n_cells - i) / larger_time_n_cells
			);
			
			// if not first or last entry, then repeat the value in the matrix
			if (i != 0)
			{
				j++;
				temporal_interpolation_matrix.add(
					j,
					0,
					(larger_time_n_cells - i) / larger_time_n_cells
				);
				temporal_interpolation_matrix.add(
					(larger_time_n_dofs-1) - j,
					1,
					(larger_time_n_cells - i) / larger_time_n_cells
				);
			}
		}

		// std::cout << "My interpolation matrix:" << std::endl;
		// temporal_interpolation_matrix.print_formatted(std::cout);
		
		// NOTE: Alternatively there is most likely also a way
		//       to get the temporal restriction/interpolation matrix in deal.II

		//////////////////////////////
		// space-time sparsity pattern
		DynamicSparsityPattern dsp(fluid_n_dofs + solid_n_dofs);
			
		//////////
		// space
		// 
		DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs(), space_dof_handler.n_dofs());
		DoFTools::make_flux_sparsity_pattern(space_dof_handler, space_dsp);

#ifdef DEBUG
		SparsityPattern space_sparsity_pattern;
		space_sparsity_pattern.copy_from(space_dsp);
		std::ofstream out_space_sparsity("space_sparsity_pattern.svg");
		space_sparsity_pattern.print_svg(out_space_sparsity);
#endif

		//////////
		// time
		//

		// fluid
		DynamicSparsityPattern fluid_time_dsp(slab->fluid_time_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(slab->fluid_time_dof_handler, fluid_time_dsp);
		SparsityPattern fluid_time_sparsity_pattern; // NOTE: This temporal sparsity pattern is only to compute the interface sparsity pattern. Therefore, we do not need jump terms here.
		fluid_time_sparsity_pattern.copy_from(fluid_time_dsp);
		// include jump terms in temporal sparsity pattern
		// for Gauss-Legendre quadrature we need to couple all temporal DoFs between two neighboring time intervals
		unsigned int fluid_time_block_size = slab->fluid_time_fe.degree + 1;
		for (unsigned int k = 1; k < slab->fluid_time_triangulation.n_active_cells(); ++k)
		for (unsigned int ii = 0; ii < fluid_time_block_size; ++ii)
			for (unsigned int jj = 0; jj < fluid_time_block_size; ++jj)
				fluid_time_dsp.add(k*fluid_time_block_size+ii, (k-1)*fluid_time_block_size+jj);
		
		// add space-time sparsity pattern for (fluid,fluid)-block
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() < fluid_n_space_dofs) && (space_entry.column() < fluid_n_space_dofs))
			for (auto &time_entry : fluid_time_dsp)
				dsp.add(
					space_entry.row()    + fluid_n_space_dofs * time_entry.row(),	// test  function
					space_entry.column() + fluid_n_space_dofs * time_entry.column() // trial function
				);

		// solid
		DynamicSparsityPattern solid_time_dsp(slab->solid_time_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(slab->solid_time_dof_handler, solid_time_dsp);
		SparsityPattern solid_time_sparsity_pattern; // NOTE: This temporal sparsity pattern is only to compute the interface sparsity pattern. Therefore, we do not need jump terms here.
		solid_time_sparsity_pattern.copy_from(solid_time_dsp);
		// include jump terms in temporal sparsity pattern
		// for Gauss-Legendre quadrature we need to couple all temporal DoFs between two neighboring time intervals
		unsigned int solid_time_block_size = slab->solid_time_fe.degree + 1;
		for (unsigned int k = 1; k < slab->solid_time_triangulation.n_active_cells(); ++k)
		for (unsigned int ii = 0; ii < solid_time_block_size; ++ii)
			for (unsigned int jj = 0; jj < solid_time_block_size; ++jj)
			solid_time_dsp.add(k*solid_time_block_size+ii, (k-1)*solid_time_block_size+jj);

		// add space-time sparsity pattern for (solid,solid)-block 
		// Note: offset row and column entries by number of fluid DoFs
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() >= fluid_n_space_dofs) && (space_entry.column() >= fluid_n_space_dofs))
			for (auto &time_entry : solid_time_dsp)
				dsp.add(
					(space_entry.row()-fluid_n_space_dofs)    + solid_n_space_dofs * time_entry.row()    + fluid_n_dofs,   // test  function
					(space_entry.column()-fluid_n_space_dofs) + solid_n_space_dofs * time_entry.column() + fluid_n_dofs    // trial function
				);

		// interface terms:

		// fluid-solid
		DynamicSparsityPattern fluid_solid_time_dsp(
			slab->fluid_time_dof_handler.n_dofs(),
			slab->solid_time_dof_handler.n_dofs()
		);
		if (slab->fluid_time_dof_handler.n_dofs() <= slab->solid_time_dof_handler.n_dofs())
			fluid_solid_time_dsp.compute_Tmmult_pattern(
				temporal_interpolation_sparsity_pattern,
				solid_time_sparsity_pattern
			);
		else
			fluid_solid_time_dsp.compute_mmult_pattern(
				fluid_time_sparsity_pattern,
				temporal_interpolation_sparsity_pattern
			);
		
#ifdef DEBUG
		SparsityPattern fluid_solid_time_sparsity_pattern;
		fluid_solid_time_sparsity_pattern.copy_from(fluid_solid_time_dsp);
		std::ofstream out_fluid_solid_time_sparsity("fluid_solid_time_sparsity_pattern.svg");
		fluid_solid_time_sparsity_pattern.print_svg(out_fluid_solid_time_sparsity);
#endif

		// add space-time sparsity pattern for (fluid,solid)-block
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() < fluid_n_space_dofs) && (space_entry.column() >= fluid_n_space_dofs))
			for (auto &time_entry : fluid_solid_time_dsp)
				dsp.add(
					space_entry.row()                         + fluid_n_space_dofs * time_entry.row()    + 0,	        // test  function
					(space_entry.column()-fluid_n_space_dofs) + solid_n_space_dofs * time_entry.column() + fluid_n_dofs // trial function
				);

		// solid-fluid
		DynamicSparsityPattern solid_fluid_time_dsp(
			slab->solid_time_dof_handler.n_dofs(),
			slab->fluid_time_dof_handler.n_dofs()
		);
		if (slab->fluid_time_dof_handler.n_dofs() > slab->solid_time_dof_handler.n_dofs())
			solid_fluid_time_dsp.compute_Tmmult_pattern(
				temporal_interpolation_sparsity_pattern,
				fluid_time_sparsity_pattern
			);
		else
			solid_fluid_time_dsp.compute_mmult_pattern(
				solid_time_sparsity_pattern,
				temporal_interpolation_sparsity_pattern
			);

#ifdef DEBUG		
		SparsityPattern solid_fluid_time_sparsity_pattern;
		solid_fluid_time_sparsity_pattern.copy_from(solid_fluid_time_dsp);
		std::ofstream out_solid_fluid_time_sparsity("solid_fluid_time_sparsity_pattern.svg");
		solid_fluid_time_sparsity_pattern.print_svg(out_solid_fluid_time_sparsity);
#endif

		// add space-time sparsity pattern for (solid,fluid)-block
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() >= fluid_n_space_dofs) && (space_entry.column() < fluid_n_space_dofs))
			for (auto &time_entry : solid_fluid_time_dsp)
				dsp.add(
					(space_entry.row()-fluid_n_space_dofs) + solid_n_space_dofs * time_entry.row()    + fluid_n_dofs, // test  function
					space_entry.column()                   + fluid_n_space_dofs * time_entry.column() + 0             // trial function
				);

		////////
		// FSI
		//
		sparsity_pattern.copy_from(dsp);

#ifdef DEBUG
		std::ofstream out_sparsity("sparsity_pattern.svg");
		sparsity_pattern.print_svg(out_sparsity);
#endif	

		system_matrix.reinit(sparsity_pattern);
	}
	
	solution.reinit(fluid_n_dofs + solid_n_dofs);
	system_rhs.reinit(fluid_n_dofs + solid_n_dofs);
}

template<int dim>
void SpaceTime<dim>::assemble_system(std::shared_ptr<Slab> &slab, bool assemble_matrix) {
	system_matrix = 0;
	system_rhs    = 0;

	//////////
	// fluid
	//
	{
#ifdef DEBUG
		// check that the entries for the (fluid, fluid) block are being distributed correctly
		DynamicSparsityPattern fluid_dsp(fluid_n_dofs+solid_n_dofs, fluid_n_dofs+solid_n_dofs);
#endif

		RightHandSide<dim> right_hand_side;
		Tensor<1, dim> beta;
		if (dim == 1) {
			beta[0] = 0.;
		}
		else if (dim == 2) {
			beta[0] = 2.;
			beta[1] = 0.;
		}

		// space
		QGauss<dim> fluid_space_quad_formula(fluid_space_fe.degree + 2);
		QGauss<dim> solid_space_quad_formula(solid_space_fe.degree + 2);
		QGauss<dim-1> space_face_quad_formula(fluid_space_fe.degree + 2);

		hp::QCollection<dim> space_q_collection;
		space_q_collection.push_back(fluid_space_quad_formula);
		space_q_collection.push_back(solid_space_quad_formula);

		hp::FEValues<dim> hp_space_fe_values(space_fe_collection, space_q_collection,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int space_dofs_per_cell = fluid_space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
		
		FEFaceValues<dim>  space_fe_face_values(fluid_space_fe, space_face_quad_formula,
				update_values | update_gradients | update_normal_vectors | update_JxW_values);

		// time
		QGauss<1> time_quad_formula(slab->fluid_time_fe.degree + 2);
		FEValues<1> time_fe_values(slab->fluid_time_fe, time_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int time_dofs_per_cell = slab->fluid_time_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
		std::vector<types::global_dof_index> time_prev_local_dof_indices(time_dofs_per_cell);

		// time FEValues for t_m^+ on current time interval I_m
		FEValues<1> time_fe_face_values(slab->fluid_time_fe, Quadrature<1>({Point<1>(0.)}), update_values); // using left box rule quadrature
		// time FEValues for t_m^- on previous time interval I_{m-1}
		FEValues<1> time_prev_fe_face_values(slab->fluid_time_fe, Quadrature<1>({Point<1>(1.)}), update_values); // using right box rule quadrature
	
		// local contributions on space-time cell
		FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		FullMatrix<double> cell_jump(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

		// locally assemble on each space-time cell
		for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
		  if (space_cell->material_id() == fluid_domain_id)
		  {
			hp_space_fe_values.reinit(space_cell);

			const FEValues<dim> &space_fe_values = hp_space_fe_values.get_present_fe_values();

			space_cell->get_dof_indices(space_local_dof_indices);
			double h = space_cell->diameter();

			for (const auto &time_cell : slab->fluid_time_dof_handler.active_cell_iterators()) {
				time_fe_values.reinit(time_cell);
				time_cell->get_dof_indices(time_local_dof_indices);
				
				cell_matrix = 0;
				cell_rhs = 0;
				cell_jump = 0;

				for (const unsigned int qq : time_fe_values.quadrature_point_indices())
				{
					// time quadrature point
					const double t_qq = time_fe_values.quadrature_point(qq)[0];
					right_hand_side.set_time(t_qq);

					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						// space quadrature point
						const auto x_q = space_fe_values.quadrature_point(q);

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								// right hand side
								cell_rhs(i + ii * space_dofs_per_cell) += (space_fe_values[fluid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
																		   right_hand_side.value(x_q, fluid_space_fe.system_to_component_index(i).first) *    // g(t_qq, x_q)
																		   space_fe_values.JxW(q) * time_fe_values.JxW(qq)								   // d(t,x)
								);

								// system matrix
								if (assemble_matrix)
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_matrix(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												space_fe_values[fluid_velocity].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *	// ∂_t ϕ^v_{i,ii}(t_qq, x_q)
												space_fe_values[fluid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)		//     ϕ^v_{j,jj}(t_qq, x_q)
																																	// +
												+ nu * space_fe_values[fluid_velocity].gradient(i, q) * time_fe_values.shape_value(ii, qq) * // ν * ∇_x ϕ^v_{i,ii}(t_qq, x_q)
												space_fe_values[fluid_velocity].gradient(j, q) * time_fe_values.shape_value(jj, qq)		     //     ∇_x ϕ^v_{j,jj}(t_qq, x_q)
																																	// +
												+ beta * space_fe_values[fluid_velocity].gradient(i, q) * time_fe_values.shape_value(ii, qq) * // β · ∇_x ϕ^v_{i,ii}(t_qq, x_q)
												space_fe_values[fluid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)               //         ϕ^v_{j,jj}(t_qq, x_q)
																																	// +
												+ space_fe_values[fluid_displacement].gradient(i, q) * time_fe_values.shape_value(ii, qq) *	 // ∇_x ϕ^u_{i,ii}(t_qq, x_q)
												space_fe_values[fluid_displacement].gradient(j, q) * time_fe_values.shape_value(jj, qq)	     // ∇_x ϕ^u_{j,jj}(t_qq, x_q)
											) *
											space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)
							}
					}
				}

				// assemble jump terms in system matrix and intial condition in RHS
				// jump terms: ([v]_m,φ_m^+)_Ω = (v_m^+,φ_m^+)_Ω - (v_m^-,φ_m^+)_Ω = (A) - (B)
				time_fe_face_values.reinit(time_cell);
				
				// first we assemble (A): (v_m^+,φ_m^+)_Ω
				if (assemble_matrix)
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
										cell_matrix(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell
										) += (
											space_fe_values[fluid_velocity].value(i, q) * time_fe_face_values.shape_value(ii, 0) * //  ϕ^v_{i,ii}(t_m^+, x_q)
											space_fe_values[fluid_velocity].value(j, q) * time_fe_face_values.shape_value(jj, 0)   //  ϕ^v_{j,jj}(t_m^+, x_q)
										) * space_fe_values.JxW(q); 						//  d(x)

				// initial condition and jump terms
				if (time_cell->active_cell_index() == 0)
				{
					//////////////////////////
					// initial condition

					// (v_0^-,φ_0^-)_Ω
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						double initial_solution_v_x_q = 0.;
						for (const unsigned int j : space_fe_values.dof_indices())
						{
							initial_solution_v_x_q += initial_solution_fluid[space_local_dof_indices[j]] * space_fe_values[fluid_velocity].value(j, q);
						}

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								cell_rhs(i + ii * space_dofs_per_cell) += (initial_solution_v_x_q *										   // v0(x_q)
																		space_fe_values[fluid_velocity].value(i, q) * time_fe_face_values.shape_value(ii, 0) * // ϕ^v_{i,ii}(0^+, x_q)
																		space_fe_values.JxW(q)															// d(x)
								);
							}
					}
				}
				else
				{
					//////////////
					// jump term

					// now we assemble (B): - (u_m^-,φ_m^+)_Ω
					// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre quadrature in time
					if (assemble_matrix)
						for (const unsigned int q : space_fe_values.quadrature_point_indices())
							for (const unsigned int i : space_fe_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_jump(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												-1. * space_fe_values[fluid_velocity].value(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -ϕ^v_{i,ii}(t_m^-, x_q)
												space_fe_values[fluid_velocity].value(j, q) * time_fe_face_values.shape_value(jj, 0)			  //  ϕ^v_{j,jj}(t_m^+, x_q)
											) * space_fe_values.JxW(q); //  d(x)
				}

				// distribute local to global
				for (const unsigned int i : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
					{
						// right hand side
						system_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs) += cell_rhs(i + ii * space_dofs_per_cell);

						// system matrix
						if (assemble_matrix)
							for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
								{
									system_matrix.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
										space_local_dof_indices[j] + time_local_dof_indices[jj] * fluid_n_space_dofs,
										cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));
#ifdef DEBUG
									fluid_dsp.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
										space_local_dof_indices[j] + time_local_dof_indices[jj] * fluid_n_space_dofs
									);
#endif
								}
					}

				// distribute cell jump
				if (assemble_matrix)
					if (time_cell->active_cell_index() > 0)
						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										system_matrix.add(
											space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
											space_local_dof_indices[j] + time_prev_local_dof_indices[jj] * fluid_n_space_dofs,
											cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

#ifdef DEBUG
										fluid_dsp.add(
											space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
											space_local_dof_indices[j] + time_prev_local_dof_indices[jj] * fluid_n_space_dofs
										);
#endif
									}

				// prepare next time cell
				if (time_cell->active_cell_index() < slab->fluid_time_triangulation.n_active_cells() - 1)
				{
					time_prev_fe_face_values.reinit(time_cell);
					time_cell->get_dof_indices(time_prev_local_dof_indices);
				}
			}

			// interface terms for (fluid,fluid)
			if (assemble_matrix)
				for (const unsigned int space_face : space_cell->face_indices())
					if (space_cell->at_boundary(space_face) == false) // face is not at boundary
					if (space_cell->neighbor(space_face)->material_id() == solid_domain_id) // face is at interface (= fluid & solid cell meet)
					{
						space_fe_face_values.reinit(space_cell, space_face);
						for (const auto &time_cell : slab->fluid_time_dof_handler.active_cell_iterators()) {
							time_fe_values.reinit(time_cell);
							time_cell->get_dof_indices(time_local_dof_indices);

							cell_matrix = 0;

							for (const unsigned int qq : time_fe_values.quadrature_point_indices())
							for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
								for (const unsigned int i : space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_face_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
										cell_matrix(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell
										) += (
											-nu *                                                                                                                     // -ν
											space_fe_face_values[fluid_velocity].gradient(i, q) * space_fe_face_values.normal_vector(q) * time_fe_values.shape_value(ii, qq) * // ∇_x ϕ^v_{i,ii}(t_qq, x_q) · n_f(x_q)
											space_fe_face_values[fluid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)                                              //     ϕ^v_{j,jj}(t_qq, x_q)
																																									// +
											-1. * 																	  												         // -1
											space_fe_face_values[fluid_displacement].gradient(i, q) * space_fe_face_values.normal_vector(q) * time_fe_values.shape_value(ii, qq) * // ∇_x ϕ^u_{i,ii}(t_qq, x_q) · n_f(x_q)
											space_fe_face_values[fluid_displacement].value(j, q) * time_fe_values.shape_value(jj, qq)                                              //     ϕ^u_{j,jj}(t_qq, x_q)
																																									// +
											+ (gamma * nu / h) *                                                            // γν/h
											space_fe_face_values[fluid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ^v_{i,ii}(t_qq, x_q)
											space_fe_face_values[fluid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)   // ϕ^v_{j,jj}(t_qq, x_q)
																														// +
											+ (gamma / h) * 																	  // γ/h
											space_fe_face_values[fluid_displacement].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ^u_{i,ii}(t_qq, x_q)
											space_fe_face_values[fluid_displacement].value(j, q) * time_fe_values.shape_value(jj, qq)   // ϕ^u_{j,jj}(t_qq, x_q)
										) * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							// d(t,x)

							// distribute local to global
							for (const unsigned int i : space_fe_face_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
								{
									system_matrix.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
										space_local_dof_indices[j] + time_local_dof_indices[jj] * fluid_n_space_dofs,
										cell_matrix(
											i + ii * space_dofs_per_cell,
											j + jj * space_dofs_per_cell
										)
									);

#ifdef DEBUG
									fluid_dsp.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
										space_local_dof_indices[j] + time_local_dof_indices[jj] * fluid_n_space_dofs
									);
#endif
								}
						}
					}
		  }
		}

#ifdef DEBUG
		if (assemble_matrix)
		{
			SparsityPattern fluid_sparsity_pattern;
			fluid_sparsity_pattern.copy_from(fluid_dsp);
			std::ofstream out_fluid_sparsity("fluid_sparsity_pattern.svg");
			fluid_sparsity_pattern.print_svg(out_fluid_sparsity);
		}
#endif
	}
	
	//////////
	// solid
	//
	{
#ifdef DEBUG
		// check that the entries for the (solid, solid) block are being distributed correctly
		DynamicSparsityPattern solid_dsp(fluid_n_dofs+solid_n_dofs, fluid_n_dofs+solid_n_dofs);
#endif

		RightHandSide<dim> right_hand_side;

		// space
		QGauss<dim> fluid_space_quad_formula(fluid_space_fe.degree + 2);
		QGauss<dim> solid_space_quad_formula(solid_space_fe.degree + 2);
		QGauss<dim-1> space_face_quad_formula(solid_space_fe.degree + 2);

		hp::QCollection<dim> space_q_collection;
		space_q_collection.push_back(fluid_space_quad_formula);
		space_q_collection.push_back(solid_space_quad_formula);

		hp::FEValues<dim> hp_space_fe_values(space_fe_collection, space_q_collection,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int space_dofs_per_cell = solid_space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
		
		FEFaceValues<dim>  space_fe_face_values(solid_space_fe, space_face_quad_formula,
				update_values | update_gradients | update_normal_vectors | update_JxW_values);

		// time
		QGauss<1> time_quad_formula(slab->solid_time_fe.degree + 2);
		FEValues<1> time_fe_values(slab->solid_time_fe, time_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int time_dofs_per_cell = slab->solid_time_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
		std::vector<types::global_dof_index> time_prev_local_dof_indices(time_dofs_per_cell);

		// time FEValues for t_m^+ on current time interval I_m
		FEValues<1> time_fe_face_values(slab->solid_time_fe, Quadrature<1>({Point<1>(0.)}), update_values); // using left box rule quadrature
		// time FEValues for t_m^- on previous time interval I_{m-1}
		FEValues<1> time_prev_fe_face_values(slab->solid_time_fe, Quadrature<1>({Point<1>(1.)}), update_values); // using right box rule quadrature

		// local contributions on space-time cell
		FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		FullMatrix<double> cell_jump(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

		// locally assemble on each space-time cell
		for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
			if (space_cell->material_id() == solid_domain_id)
			{
				hp_space_fe_values.reinit(space_cell);

				const FEValues<dim> &space_fe_values = hp_space_fe_values.get_present_fe_values();

				space_cell->get_dof_indices(space_local_dof_indices);
				for (const auto &time_cell : slab->solid_time_dof_handler.active_cell_iterators()) {
					time_fe_values.reinit(time_cell);
					time_cell->get_dof_indices(time_local_dof_indices);
					
					cell_matrix = 0;
					cell_rhs = 0;
					cell_jump = 0;

					for (const unsigned int qq : time_fe_values.quadrature_point_indices())
					{
						// time quadrature point
						const double t_qq = time_fe_values.quadrature_point(qq)[0];
						right_hand_side.set_time(t_qq);

						for (const unsigned int q : space_fe_values.quadrature_point_indices())
						{
							// space quadrature point
							const auto x_q = space_fe_values.quadrature_point(q);

							for (const unsigned int i : space_fe_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
								{
									// right hand side
									cell_rhs(i + ii * space_dofs_per_cell) += (
										space_fe_values[solid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
										right_hand_side.value(x_q, solid_space_fe.system_to_component_index(i).first) *	   // g(t_qq, x_q)
										space_fe_values.JxW(q) * time_fe_values.JxW(qq)								// d(t,x)
									);

									// system matrix
									if (assemble_matrix)
										for (const unsigned int j : space_fe_values.dof_indices())
											for (const unsigned int jj : time_fe_values.dof_indices())
												cell_matrix(
													j + jj * space_dofs_per_cell,
													i + ii * space_dofs_per_cell
												) += (
													space_fe_values[solid_velocity].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *		// ∂_t ϕ^v_{i,ii}(t_qq, x_q)
													space_fe_values[solid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)			//     ϕ^v_{j,jj}(t_qq, x_q)
																																	// +
													+ lambda * space_fe_values[solid_displacement].gradient(i, q) * time_fe_values.shape_value(ii, qq) * // λ * ∇_x ϕ^u_{i,ii}(t_qq, x_q)
													space_fe_values[solid_velocity].gradient(j, q) * time_fe_values.shape_value(jj, qq)		      	     //     ∇_x ϕ^v_{j,jj}(t_qq, x_q)
																																	// +
													+ delta * space_fe_values[solid_velocity].gradient(i, q) * time_fe_values.shape_value(ii, qq) *      // δ * ∇_x ϕ^v_{i,ii}(t_qq, x_q)
													space_fe_values[solid_velocity].gradient(j, q) * time_fe_values.shape_value(jj, qq)                  //     ∇_x ϕ^v_{j,jj}(t_qq, x_q)
																																		// +
													+ space_fe_values[solid_displacement].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *	// ∂_t ϕ^u_{i,ii}(t_qq, x_q)
													space_fe_values[solid_displacement].value(j, q) * time_fe_values.shape_value(jj, qq)		//     ϕ^u_{j,jj}(t_qq, x_q)
																																		// -
													- space_fe_values[solid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) *			//  ϕ^v_{i,ii}(t_qq, x_q)
													space_fe_values[solid_displacement].value(j, q) * time_fe_values.shape_value(jj, qq)			//  ϕ^u_{j,jj}(t_qq, x_q)
												) * space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)
								}
						}
					}

					// assemble jump terms in system matrix and intial condition in RHS
					// jump terms: 
					//    a) for displacement: ([u]_m,ϕ^{u,+}_m)_Ω = (u_m^+,ϕ^{u,+}_m)_Ω - (u_m^-,ϕ^{u,+}_m)_Ω = (A1) - (B1)
					//    b) for velocity:     ([v]_m,ϕ^{v,+}_m)_Ω = (v_m^+,ϕ^{v,+}_m)_Ω - (v_m^-,ϕ^{v,+}_m)_Ω = (A2) - (B2)
					time_fe_face_values.reinit(time_cell);

					// first we assemble (A1) and (A2): (u_m^+,ϕ^{u,+}_m)_Ω and (v_m^+,ϕ^{v,+}_m)_Ω
					if (assemble_matrix)
						for (const unsigned int q : space_fe_values.quadrature_point_indices())
							for (const unsigned int i : space_fe_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_matrix(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												space_fe_values[solid_displacement].value(i, q) * time_fe_face_values.shape_value(ii, 0) * //  ϕ^u_{i,ii}(t_m^+, x_q)
												space_fe_values[solid_displacement].value(j, q) * time_fe_face_values.shape_value(jj, 0)   //  ϕ^u_{j,jj}(t_m^+, x_q)
																																// +
												+ space_fe_values[solid_velocity].value(i, q) * time_fe_face_values.shape_value(ii, 0) *  //  ϕ^v_{i,ii}(t_m^+, x_q)
												space_fe_values[solid_velocity].value(j, q) * time_fe_face_values.shape_value(jj, 0)      //  ϕ^v_{j,jj}(t_m^+, x_q)
											) * space_fe_values.JxW(q); //  d(x)

					// initial condition and jump terms
					if (time_cell->active_cell_index() == 0)
					{
						//////////////////////////
						// initial condition

						// (u_0^-,ϕ^{u,+}_0)_Ω and (v_0^-,ϕ^{v,+}_0)_Ω
						for (const unsigned int q : space_fe_values.quadrature_point_indices())
						{
							double initial_solution_u_x_q = 0.;
							double initial_solution_v_x_q = 0.;
							for (const unsigned int j : space_fe_values.dof_indices())
							{
								initial_solution_u_x_q += initial_solution_solid[space_local_dof_indices[j]] * space_fe_values[solid_displacement].value(j, q);
								initial_solution_v_x_q += initial_solution_solid[space_local_dof_indices[j]] * space_fe_values[solid_velocity].value(j, q);
							}
							
							for (const unsigned int i : space_fe_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
								{
									cell_rhs(i + ii * space_dofs_per_cell) += (
										initial_solution_u_x_q *                                             // u_0(x_q)
										space_fe_values[solid_displacement].value(i, q) * time_fe_face_values.shape_value(ii, 0) // ϕ^u_{i,ii}(0^+, x_q)
																												// +
										+ initial_solution_v_x_q *                                       // v_0(x_q)
										space_fe_values[solid_velocity].value(i, q) * time_fe_face_values.shape_value(ii, 0) // ϕ^v_{i,0}(0, x_q)
									) * space_fe_values.JxW(q);   // d(x)
								}
						}
					}
					else
					{
						//////////////
						// jump term

						// now we assemble (B1) and (B2): - (u_m^-,ϕ^{u,+}_m)_Ω and - (v_m^-,ϕ^{v,+}_m)_Ω
						// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre quadrature in time
						if (assemble_matrix)
							for (const unsigned int q : space_fe_values.quadrature_point_indices())
								for (const unsigned int i : space_fe_values.dof_indices())
									for (const unsigned int ii : time_fe_values.dof_indices())
										for (const unsigned int j : space_fe_values.dof_indices())
											for (const unsigned int jj : time_fe_values.dof_indices())
												cell_jump(
													j + jj * space_dofs_per_cell,
													i + ii * space_dofs_per_cell
												) += (
													-1. * space_fe_values[solid_displacement].value(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -ϕ^u_{i,ii}(t_m^-, x_q)
													space_fe_values[solid_displacement].value(j, q) * time_fe_face_values.shape_value(jj, 0)              //  ϕ^u_{j,jj}(t_m^+, x_q)
													// +
													- space_fe_values[solid_velocity].value(i, q)   * time_prev_fe_face_values.shape_value(ii, 0) *  // -ϕ^v_{i,ii}(t_m^-, x_q)
													space_fe_values[solid_velocity].value(j, q) * time_fe_face_values.shape_value(jj, 0)             //  ϕ^v_{j,jj}(t_m^+, x_q)
												) * space_fe_values.JxW(q); 		      //  d(x)
					}

					// distribute local to global (NOTE: need to offset rows and columns by number of fluid space-time DoFs)
					for (const unsigned int i : space_fe_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
						{
							// right hand side
							system_rhs((space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs) += cell_rhs(i + ii * space_dofs_per_cell);

							// system matrix
							if (assemble_matrix)
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										system_matrix.add(
											(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
											(space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs,
											cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

#ifdef DEBUG
										solid_dsp.add(
											(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
											(space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs
										);
#endif
									}
						}

					// distribute cell jump (NOTE: need to offset rows and columns by number of fluid space-time DoFs)
					if (assemble_matrix)
						if (time_cell->active_cell_index() > 0)
							for (const unsigned int i : space_fe_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
										{
											system_matrix.add(
												(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs      + fluid_n_dofs,
												(space_local_dof_indices[j]-fluid_n_space_dofs) + time_prev_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs,
												cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

#ifdef DEBUG
											solid_dsp.add(
												(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs      + fluid_n_dofs,
												(space_local_dof_indices[j]-fluid_n_space_dofs) + time_prev_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs
											);
#endif
									}

					// prepare next time cell
					if (time_cell->active_cell_index() < slab->solid_time_triangulation.n_active_cells() - 1)
					{
						time_prev_fe_face_values.reinit(time_cell);
						time_cell->get_dof_indices(time_prev_local_dof_indices);
					}
				}

				// interface terms for (solid,solid)
				if (assemble_matrix)
					for (const unsigned int space_face : space_cell->face_indices())
						if (space_cell->at_boundary(space_face) == false) // face is not at boundary
						if (space_cell->neighbor(space_face)->material_id() == fluid_domain_id) // face is at interface (= fluid & solid cell meet)
						{
							space_fe_face_values.reinit(space_cell, space_face);
							for (const auto &time_cell : slab->solid_time_dof_handler.active_cell_iterators()) {
								time_fe_values.reinit(time_cell);
								time_cell->get_dof_indices(time_local_dof_indices);

								cell_matrix = 0;

								for (const unsigned int qq : time_fe_values.quadrature_point_indices())
								for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
									for (const unsigned int i : space_fe_face_values.dof_indices())
									for (const unsigned int ii : time_fe_values.dof_indices())
										for (const unsigned int j : space_fe_face_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_matrix(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												-delta * // -δ
												space_fe_face_values[solid_velocity].gradient(i, q) * space_fe_face_values.normal_vector(q) * time_fe_values.shape_value(ii, qq) * // ∇_x ϕ^v_{i,ii}(t_qq, x_q) · n_s(x_q)
												space_fe_face_values[solid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)                                              //     ϕ^v_{j,jj}(t_qq, x_q)
											) * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							// d(t,x)

								// distribute local to global (NOTE: need to offset rows and columns by number of fluid space-time DoFs)
								for (const unsigned int i : space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										system_matrix.add(
											(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
											(space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs,
											cell_matrix(
												i + ii * space_dofs_per_cell,
												j + jj * space_dofs_per_cell
											)
										);

#ifdef DEBUG
										solid_dsp.add(
											(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
											(space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs
										);
#endif
									}
							}
						}
			}
		}

#ifdef DEBUG
		if (assemble_matrix)
		{
			SparsityPattern solid_sparsity_pattern;
			solid_sparsity_pattern.copy_from(solid_dsp);
			std::ofstream out_solid_sparsity("solid_sparsity_pattern.svg");
			solid_sparsity_pattern.print_svg(out_solid_sparsity);
		}
#endif
	}

	//////////
	// interface
	//
	if (assemble_matrix)
	{
#ifdef DEBUG
		// create the sparsity pattern that is required for the interface terms
		DynamicSparsityPattern interface_dsp(fluid_n_dofs+solid_n_dofs, fluid_n_dofs+solid_n_dofs);
#endif

		// space:
		QGauss<dim-1> fluid_space_face_quad_formula(fluid_space_fe.degree + 2);
		QGauss<dim-1> solid_space_face_quad_formula(solid_space_fe.degree + 2);

		FEFaceValues<dim> fluid_space_fe_face_values(fluid_space_fe, fluid_space_face_quad_formula,
			update_values | update_gradients | update_normal_vectors | update_JxW_values);
		FEFaceValues<dim> solid_space_fe_face_values(solid_space_fe, solid_space_face_quad_formula,
			update_values | update_gradients | update_normal_vectors | update_JxW_values);

		const unsigned int fluid_space_dofs_per_cell = fluid_space_fe.n_dofs_per_cell();
		const unsigned int solid_space_dofs_per_cell = solid_space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> fluid_space_local_dof_indices(fluid_space_dofs_per_cell);
		std::vector<types::global_dof_index> solid_space_local_dof_indices(solid_space_dofs_per_cell);

		// time:
		//   using temporal FE with finer temporal mesh
		bool fluid_is_finer = (slab->fluid_time_dof_handler.n_dofs() > slab->solid_time_dof_handler.n_dofs());
		auto time_fe = (
			fluid_is_finer ?
				&(slab->fluid_time_fe) :
				&(slab->solid_time_fe)
		);
		auto time_dof_handler = (
			fluid_is_finer ?
				&(slab->fluid_time_dof_handler) :
				&(slab->solid_time_dof_handler)
		);
		QGauss<1> time_quad_formula(time_fe->degree + 2);
		FEValues<1> time_fe_values(*time_fe, time_quad_formula,
			update_values | update_JxW_values);
		const unsigned int time_dofs_per_cell = time_fe->n_dofs_per_cell();
		std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);

		// local contributions on space-time cell
		FullMatrix<double> cell_matrix_solid_fluid(
			solid_space_dofs_per_cell * time_dofs_per_cell,
			fluid_space_dofs_per_cell * time_dofs_per_cell
		);
		FullMatrix<double> cell_matrix_fluid_solid(
			fluid_space_dofs_per_cell * time_dofs_per_cell,
			solid_space_dofs_per_cell * time_dofs_per_cell
		);
		// std::vector<types::global_dof_index> fluid_local_dof_indices(fluid_space_dofs_per_cell * time_dofs_per_cell);
		// std::vector<types::global_dof_index> solid_local_dof_indices(solid_space_dofs_per_cell * time_dofs_per_cell);

		// locally assemble on each space-time cell
		for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
			if (space_cell->material_id() == fluid_domain_id)
		  	{
				// space_cell->get_dof_indices(fluid_space_local_dof_indices);
				double h = space_cell->diameter();

				for (const unsigned int space_face : space_cell->face_indices())
					if (space_cell->at_boundary(space_face) == false) // face is not at boundary
					if (space_cell->neighbor(space_face)->material_id() == solid_domain_id) // face is at interface (= fluid & solid cell meet)
					{
						// fluid FEFaceValues
						fluid_space_fe_face_values.reinit(space_cell, space_face);
						// get space_dof indices on this fluid space_cell (NOT on space_face!)
						space_cell->get_dof_indices(fluid_space_local_dof_indices);

						// solid FEFaceValues
						solid_space_fe_face_values.reinit(
							space_cell->neighbor(space_face),
                            space_cell->neighbor_of_neighbor(space_face)
						);
						// get space_dof indices on neighboring solid space_cell (NOT on space_face!)
						space_cell->neighbor(space_face)->get_dof_indices(solid_space_local_dof_indices);

						Assert(fluid_space_fe_face_values.n_quadrature_points == solid_space_fe_face_values.n_quadrature_points, ExcInternalError());

						for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
							time_fe_values.reinit(time_cell);
							time_cell->get_dof_indices(time_local_dof_indices);

							cell_matrix_solid_fluid = 0;
							cell_matrix_fluid_solid = 0;

							for (const unsigned int qq : time_fe_values.quadrature_point_indices())
							for (const unsigned int q : fluid_space_fe_face_values.quadrature_point_indices())
							{
								// interface terms for (fluid, solid)
								for (const unsigned int i : solid_space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : fluid_space_fe_face_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										cell_matrix_fluid_solid(
											j + jj * fluid_space_dofs_per_cell,
											i + ii * solid_space_dofs_per_cell
										) += (
											(-gamma * nu / h) *                                                            // -γν/h
											solid_space_fe_face_values[solid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ^v_{i,ii}(t_qq, x_q)
											fluid_space_fe_face_values[fluid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)   // ϕ^v_{j,jj}(t_qq, x_q)
																														// +
											+ (-gamma / h) * 																	  // -γ/h
											solid_space_fe_face_values[solid_displacement].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ^u_{i,ii}(t_qq, x_q)
											fluid_space_fe_face_values[fluid_displacement].value(j, q) * time_fe_values.shape_value(jj, qq)   // ϕ^u_{j,jj}(t_qq, x_q)
										) * fluid_space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							// d(t,x)
									}

								// interface terms for (solid, fluid)	
								for (const unsigned int i : fluid_space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : solid_space_fe_face_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										cell_matrix_solid_fluid(
											j + jj * solid_space_dofs_per_cell,
											i + ii * fluid_space_dofs_per_cell
										) += (
											nu *                                                             			                                             // ν
											fluid_space_fe_face_values[fluid_velocity].gradient(i, q) * fluid_space_fe_face_values.normal_vector(q) * time_fe_values.shape_value(ii, qq) * // ∇_x ϕ^v_{i,ii}(t_qq, x_q) · n_f(x_q)
											solid_space_fe_face_values[solid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)                                                    //     ϕ^v_{j,jj}(t_qq, x_q)
										) * fluid_space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							                                           // d(t,x)
									}
							}

							// NOTE: if the temporal interpolation matrix is very sparse, then it would be probably more efficient to
							// get the cell matrix for each spatial i,j and perform a matrix multiplication with temporal_interpolation_matrix directly,
							// since then temporal_interpolation_matrix.el(.,.) often returns zeros

							// distribute local to global for (fluid, solid)
							for (const unsigned int i : fluid_space_fe_face_values.dof_indices())
							for (const unsigned int j : solid_space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
									for (unsigned int kk = 0; kk < temporal_interpolation_matrix.n(); ++kk)
									{
#ifdef DEBUG
										if (fluid_is_finer)
											interface_dsp.add(
												fluid_space_local_dof_indices[i]                     + time_local_dof_indices[ii] * fluid_n_space_dofs + 0,
												(solid_space_local_dof_indices[j]-fluid_n_space_dofs) + kk * solid_n_space_dofs                         + fluid_n_dofs
											);
										else
											interface_dsp.add(
												fluid_space_local_dof_indices[i]                     + kk * fluid_n_space_dofs                         + 0,
												(solid_space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs
											); 
#endif

										if (fluid_is_finer)
											system_matrix.add(
												fluid_space_local_dof_indices[i]                     + time_local_dof_indices[ii] * fluid_n_space_dofs + 0,
												(solid_space_local_dof_indices[j]-fluid_n_space_dofs) + kk * solid_n_space_dofs                         + fluid_n_dofs,
												cell_matrix_fluid_solid(
													i + ii * fluid_space_dofs_per_cell,
													j + jj * solid_space_dofs_per_cell
												) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk)
											);
										else
											system_matrix.add(
												fluid_space_local_dof_indices[i]                     + kk * fluid_n_space_dofs                         + 0,
												(solid_space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs,
												cell_matrix_fluid_solid(
													i + ii * fluid_space_dofs_per_cell,
													j + jj * solid_space_dofs_per_cell
												) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk)
											); 
									}

							// distribute local to global for (solid, fluid)
							for (const unsigned int i : solid_space_fe_face_values.dof_indices())
							for (const unsigned int j : fluid_space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
									for (unsigned int kk = 0; kk < temporal_interpolation_matrix.n(); ++kk)
									{
#ifdef DEBUG
										if (!fluid_is_finer)
											interface_dsp.add(
												(solid_space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
												fluid_space_local_dof_indices[j]                      + kk * fluid_n_space_dofs                         + 0
											);
										else
											interface_dsp.add(
												(solid_space_local_dof_indices[i]-fluid_n_space_dofs) + kk * solid_n_space_dofs                         + fluid_n_dofs,
												fluid_space_local_dof_indices[j]                      + time_local_dof_indices[jj] * fluid_n_space_dofs + 0
											);
#endif

										if (!fluid_is_finer)
											system_matrix.add(
												(solid_space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
												fluid_space_local_dof_indices[j]                      + kk * fluid_n_space_dofs                         + 0,
												cell_matrix_solid_fluid(
													i + ii * solid_space_dofs_per_cell,
													j + jj * fluid_space_dofs_per_cell
												) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk)
											);
										else
											system_matrix.add(
												(solid_space_local_dof_indices[i]-fluid_n_space_dofs) + kk * solid_n_space_dofs                         + fluid_n_dofs,
												fluid_space_local_dof_indices[j]                      + time_local_dof_indices[jj] * fluid_n_space_dofs + 0,
												cell_matrix_solid_fluid(
													i + ii * solid_space_dofs_per_cell,
													j + jj * fluid_space_dofs_per_cell
												) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk)
											);
									}
						}
					}
			}
		}

#ifdef DEBUG
		SparsityPattern interface_sparsity_pattern;
		interface_sparsity_pattern.copy_from(interface_dsp);
		std::ofstream out_interface_sparsity("interface_sparsity_pattern.svg");
		interface_sparsity_pattern.print_svg(out_interface_sparsity);
#endif
	}

	apply_boundary_conditions(slab);
}	

template<int dim>
void SpaceTime<dim>::apply_boundary_conditions(std::shared_ptr<Slab> &slab) {
   //////////
   // fluid
   //
   {
	   // apply the spatial Dirichlet boundary conditions at each temporal DoF
	   BoundaryValues<dim> boundary_func;
	   
	   // remove old temporal support points
	   fluid_time_support_points.clear();
	   
	   FEValues<1> time_fe_values(
			slab->fluid_time_fe, 
			Quadrature<1>(slab->fluid_time_fe.get_unit_support_points()),
			update_quadrature_points
		);
	   std::vector<types::global_dof_index> time_local_dof_indices(slab->fluid_time_fe.n_dofs_per_cell());

	   for (const auto &time_cell : slab->fluid_time_dof_handler.active_cell_iterators())
	   {
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			// using temporal support points as quadrature points
			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				boundary_func.set_time(t_qq);
				fluid_time_support_points.insert(std::make_pair(t_qq, time_local_dof_indices[qq]));

				// determine spatial boundary values at temporal support point
				std::map<types::global_dof_index, double> boundary_values;
				VectorTools::interpolate_boundary_values(space_dof_handler, 0, boundary_func, boundary_values, space_fe_collection.component_mask(fluid_displacement));
				VectorTools::interpolate_boundary_values(space_dof_handler, 0, boundary_func, boundary_values, space_fe_collection.component_mask(fluid_velocity));

				// calculate the correct space-time entry and apply the Dirichlet BC
				for (auto &entry : boundary_values)
				{
					types::global_dof_index id = entry.first + time_local_dof_indices[qq] * fluid_n_space_dofs;

					// apply BC
					for (typename SparseMatrix<double>::iterator p = system_matrix.begin(id); p != system_matrix.end(id); ++p)
						p->value() = 0.;
					system_matrix.set(id, id, 1.);
					system_rhs(id) = entry.second;
				}
			}
	   }
   }

   //////////
   // solid
   //
   {
	   // apply the spatial Dirichlet boundary conditions at each temporal DoF
	   BoundaryValues<dim> boundary_func;

	   // remove old temporal support points
	   solid_time_support_points.clear();

	   FEValues<1> time_fe_values(
			slab->solid_time_fe, 
			Quadrature<1>(slab->solid_time_fe.get_unit_support_points()), 
			update_quadrature_points
		);
	   std::vector<types::global_dof_index> time_local_dof_indices(slab->solid_time_fe.n_dofs_per_cell());

	   for (const auto &time_cell : slab->solid_time_dof_handler.active_cell_iterators())
	   {
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			// using temporal support points as quadrature points
			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				boundary_func.set_time(t_qq);
				solid_time_support_points.insert(std::make_pair(t_qq, time_local_dof_indices[qq]));

				// determine spatial boundary values at temporal support point
				std::map<types::global_dof_index, double> boundary_values;
				VectorTools::interpolate_boundary_values(space_dof_handler, 0, boundary_func, boundary_values, space_fe_collection.component_mask(solid_displacement));
				VectorTools::interpolate_boundary_values(space_dof_handler, 0, boundary_func, boundary_values, space_fe_collection.component_mask(solid_velocity));

				// calculate the correct space-time entry and apply the Dirichlet BC
				for (auto &entry : boundary_values)
				{
					// (NOTE: remember that solid block is offset by number of fluid space-time DoFs)
					types::global_dof_index id = (entry.first-fluid_n_space_dofs) + time_local_dof_indices[qq] * solid_n_space_dofs + fluid_n_dofs;

					// apply BC
					for (typename SparseMatrix<double>::iterator p = system_matrix.begin(id); p != system_matrix.end(id); ++p)
						p->value() = 0.;
					system_matrix.set(id, id, 1.);
					system_rhs(id) = entry.second;
				}
			}
	   }
   }
}

template<int dim>
void SpaceTime<dim>::solve(bool invert) {
	if (invert)
		A_direct.initialize(system_matrix);
	A_direct.vmult(solution, system_rhs);
}

template<int dim>
void SpaceTime<dim>::compute_goal_functional(std::shared_ptr<Slab> &slab) {
	std::vector<Vector<double>> solution_at_t_qq(
		std::max(slab->fluid_time_dof_handler.n_dofs(), slab->solid_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
		Vector<double>(space_dof_handler.n_dofs()) // joint solution at timepoint t_qq
	);
	std::vector<double> fluid_values_t_qq;
	std::vector<double> solid_values_t_qq;

	// fill the solutions on finer temporal mesh
	get_solution_on_finer_mesh(slab, solution_at_t_qq, fluid_values_t_qq, solid_values_t_qq);
	
	bool fluid_is_finer = (slab->fluid_time_dof_handler.n_dofs() > slab->solid_time_dof_handler.n_dofs());
	auto time_fe = (
		fluid_is_finer ?
			&(slab->fluid_time_fe) :
			&(slab->solid_time_fe)
	);
	auto time_dof_handler = (
		fluid_is_finer ?
			&(slab->fluid_time_dof_handler) :
			&(slab->solid_time_dof_handler)
	);
	FEValues<1> time_fe_values(*time_fe, QGauss<1>(time_fe->degree+4), update_values | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> time_local_dof_indices(time_fe->n_dofs_per_cell());
	
	QGauss<dim> fluid_space_quad_formula(fluid_space_fe.degree+8);
	QGauss<dim> solid_space_quad_formula(solid_space_fe.degree+8);
	
	hp::QCollection<dim> space_q_collection;
	space_q_collection.push_back(fluid_space_quad_formula);
	space_q_collection.push_back(solid_space_quad_formula);

	hp::FEValues<dim> hp_space_fe_values(space_fe_collection, space_q_collection,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
	Solution<dim> solution_func;

	DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs(), space_dof_handler.n_dofs());
	DoFTools::make_flux_sparsity_pattern(space_dof_handler, space_dsp);
	SparsityPattern sparsity;
	sparsity.copy_from(space_dsp);
	SparseMatrix<double> space_laplace_matrix(sparsity);
	MatrixCreator::create_laplace_matrix(space_dof_handler, space_q_collection, space_laplace_matrix);
	
	for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
	  time_fe_values.reinit(time_cell);
	  time_cell->get_dof_indices(time_local_dof_indices);

	  for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
	  {
	    // time quadrature point
	    double t_qq = time_fe_values.quadrature_point(qq)[0];
	    solution_func.set_time(t_qq);
	    
	    // get the space solution at the quadrature point
	    Vector<double> space_solution(space_dof_handler.n_dofs());
	    for (const unsigned int ii : time_fe_values.dof_indices())
	    {
		  if (!SOLID_SOURCE)
		  {
			// only consider the solution at v_f
			for (unsigned int i = fluid_n_space_u; i < fluid_n_space_u + fluid_n_space_v; ++i)
				space_solution(i) += solution_at_t_qq[time_local_dof_indices[ii]](i) * time_fe_values.shape_value(ii, qq);
		  }
		  else
		  {
			// only consider the solution at u_s
			for (unsigned int i = fluid_n_space_u + fluid_n_space_v; i < fluid_n_space_u + fluid_n_space_v + solid_n_space_u; ++i)
				space_solution(i) += solution_at_t_qq[time_local_dof_indices[ii]](i) * time_fe_values.shape_value(ii, qq);
		  }
	    }

	    // compute_global_error by hand
		//     error(t_qq) = e * M * e
		Vector<double> tmp(space_dof_handler.n_dofs());
		space_laplace_matrix.vmult(tmp, space_solution);
		double goal_func_t_qq = 0.;
		if (!SOLID_SOURCE)
			goal_func_t_qq = nu * (tmp * space_solution);
		else
			goal_func_t_qq = lambda * (tmp * space_solution);
	    
	    // add local contributions to global L2 error
	    goal_func_value +=  goal_func_t_qq * time_fe_values.JxW(qq);
	  }
	}
}

template<int dim>
void SpaceTime<dim>::process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab) {
	std::vector<Vector<double>> solution_at_t_qq(
		std::max(slab->fluid_time_dof_handler.n_dofs(), slab->solid_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
		Vector<double>(space_dof_handler.n_dofs()) // joint solution at timepoint t_qq
	);
	std::vector<double> fluid_values_t_qq;
	std::vector<double> solid_values_t_qq;

	// fill the solutions on finer temporal mesh
	get_solution_on_finer_mesh(slab, solution_at_t_qq, fluid_values_t_qq, solid_values_t_qq);
	
	bool fluid_is_finer = (slab->fluid_time_dof_handler.n_dofs() > slab->solid_time_dof_handler.n_dofs());
	auto time_fe = (
		fluid_is_finer ?
			&(slab->fluid_time_fe) :
			&(slab->solid_time_fe)
	);
	auto time_dof_handler = (
		fluid_is_finer ?
			&(slab->fluid_time_dof_handler) :
			&(slab->solid_time_dof_handler)
	);
	FEValues<1> time_fe_values(*time_fe, QGauss<1>(time_fe->degree+4), update_values | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> time_local_dof_indices(time_fe->n_dofs_per_cell());
	
	QGauss<dim> fluid_space_quad_formula(fluid_space_fe.degree+8);
	QGauss<dim> solid_space_quad_formula(solid_space_fe.degree+8);
	
	hp::QCollection<dim> space_q_collection;
	space_q_collection.push_back(fluid_space_quad_formula);
	space_q_collection.push_back(solid_space_quad_formula);

	hp::FEValues<dim> hp_space_fe_values(space_fe_collection, space_q_collection,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
	Solution<dim> solution_func;

	DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs(), space_dof_handler.n_dofs());
	DoFTools::make_flux_sparsity_pattern(space_dof_handler, space_dsp);
	SparsityPattern sparsity;
	sparsity.copy_from(space_dsp);
	SparseMatrix<double> space_mass_matrix(sparsity);
	MatrixCreator::create_mass_matrix(space_dof_handler, space_q_collection, space_mass_matrix);
	
	for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
	  time_fe_values.reinit(time_cell);
	  time_cell->get_dof_indices(time_local_dof_indices);

	  for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
	  {
	    // time quadrature point
	    double t_qq = time_fe_values.quadrature_point(qq)[0];
	    solution_func.set_time(t_qq);
	    
	    // get the space solution at the quadrature point
	    Vector<double> space_solution(space_dof_handler.n_dofs());
	    for (const unsigned int ii : time_fe_values.dof_indices())
	    {
	      for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) += solution_at_t_qq[time_local_dof_indices[ii]](i) * time_fe_values.shape_value(ii, qq);
	    }
	     	      
	    // compute the L2 error at the temporal quadrature point
		Vector<double> analytical_solution(space_dof_handler.n_dofs());
		VectorTools::interpolate(space_dof_handler,
					solution_func,
					analytical_solution,
					ComponentMask());
			
		// e = u - u_h
		analytical_solution.add(-1., space_solution);

		// split error into fluid and solid part
		Vector<double> fluid_error(space_dof_handler.n_dofs());
		Vector<double> solid_error(space_dof_handler.n_dofs());
		for (unsigned int i = 0; i < fluid_n_space_u + fluid_n_space_v; ++i)
			fluid_error(i) = analytical_solution(i);
		for (unsigned int i = fluid_n_space_u + fluid_n_space_v; i < fluid_n_space_u + fluid_n_space_v + solid_n_space_u + solid_n_space_v; ++i)
			solid_error(i) = analytical_solution(i);

	    // compute_global_error by hand
		//     error(t_qq) = e * M * e
		Vector<double> tmp(space_dof_handler.n_dofs());
		// fluid error t_qq
		space_mass_matrix.vmult(tmp, fluid_error);
		double L2_fluid_error_t_qq = (tmp * fluid_error);
		// solid error t_qq
		space_mass_matrix.vmult(tmp, solid_error);
		double L2_solid_error_t_qq = (tmp * solid_error);
	    
	    // add local contributions to global L2 error
	    L2_fluid_error +=  L2_fluid_error_t_qq * time_fe_values.JxW(qq);
	    L2_solid_error +=  L2_solid_error_t_qq * time_fe_values.JxW(qq);

		L2_error += (L2_fluid_error_t_qq + L2_solid_error_t_qq) * time_fe_values.JxW(qq);
	  }
	}

//	n_active_time_cells += slab->time_triangulation.n_active_cells();
//	n_time_dofs += (slab->time_dof_handler.n_dofs()-1); // first time DoF is also part of last slab

	fluid_total_n_dofs += fluid_n_dofs;
	solid_total_n_dofs += solid_n_dofs;

	if (last_slab)
	{
		L2_error = std::sqrt(L2_error);
		L2_error_vals.push_back(L2_error);

		L2_fluid_error = std::sqrt(L2_fluid_error);
		L2_solid_error = std::sqrt(L2_solid_error);
		L2_fluid_error_vals.push_back(L2_fluid_error);
		L2_solid_error_vals.push_back(L2_solid_error);

		// add values to
//		const unsigned int n_active_cells = space_triangulation.n_active_cells() * n_active_time_cells;
//		const unsigned int n_space_dofs   = space_dof_handler.n_dofs();
//		const unsigned int n_dofs         = n_space_dofs * n_time_dofs;
		const unsigned int n_dofs = fluid_total_n_dofs + solid_total_n_dofs;

		convergence_table.add_value("cycle", cycle);
//		convergence_table.add_value("cells", n_active_cells);
		convergence_table.add_value("dofs", n_dofs);
//		convergence_table.add_value("dofs(space)", n_space_dofs);
//		convergence_table.add_value("dofs(time)", n_time_dofs);
		convergence_table.add_value("L2", L2_error);
		convergence_table.add_value("L2_fluid", L2_fluid_error);
		convergence_table.add_value("L2_solid", L2_solid_error);
	}
}

template<int dim>
void SpaceTime<dim>::print_convergence_table() {
	convergence_table.set_precision("L2", 3);
	convergence_table.set_scientific("L2", true);
	convergence_table.set_precision("L2_fluid", 3);
	convergence_table.set_scientific("L2_fluid", true);
	convergence_table.set_precision("L2_solid", 3);
	convergence_table.set_scientific("L2_solid", true);
//	convergence_table.set_tex_caption("cells", "\\# cells");
	convergence_table.set_tex_caption("dofs", "\\# dofs");
//	convergence_table.set_tex_caption("dofs(space)", "\\# dofs space");
//	convergence_table.set_tex_caption("dofs(time)", "\\# dofs time");
	convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
//	convergence_table.set_tex_format("cells", "r");
	convergence_table.set_tex_format("dofs", "r");
//	convergence_table.set_tex_format("dofs(space)", "r");
//	convergence_table.set_tex_format("dofs(time)", "r");
	std::cout << std::endl;
	convergence_table.write_text(std::cout);

//	convergence_table.add_column_to_supercolumn("cycle", "n cells");
//	convergence_table.add_column_to_supercolumn("cells", "n cells");
//	std::vector<std::string> new_order;
//	new_order.emplace_back("n cells");
//	new_order.emplace_back("L2");
//	convergence_table.set_column_order(new_order);
	if (refine_space && refine_time)
	  convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);

	// compute convergence rates from 3 consecutive errors
	for (unsigned int i = 0; i < L2_error_vals.size(); ++i)
	{
	  if (i < 2)
	  convergence_table.add_value("L2...", std::string("-"));
	  else
	  {
	  double p0 = L2_error_vals[i-2];
	  double p1 = L2_error_vals[i-1];
	  double p2 = L2_error_vals[i];
	  convergence_table.add_value("L2...", std::log(std::fabs((p0-p1) / (p1-p2))) / std::log(2.));
	  }
	}

	std::cout << std::endl;
	convergence_table.write_text(std::cout);
}

template<int dim>
void SpaceTime<dim>::get_solution_on_finer_mesh(std::shared_ptr<Slab> &slab, std::vector<Vector<double>> &solution_at_t_qq, std::vector<double> &fluid_values_t_qq, std::vector<double> &solid_values_t_qq) {
	//////////
	// fluid
	//
	{
		// Iterated formula splits temporal element into many subintervals
		// Should evaluate fluid and solid at same temporal points to compare in Paraview
		int num_splits = (initial_temporal_ref_fluid >= initial_temporal_ref_solid) ? 1 : std::pow(2, initial_temporal_ref_solid);
		QIterated<1> quad(QTrapez<1>(), num_splits);
		// std::cout << "quadrature formula size: " << quad.size() << std::endl;
		FEValues<1> time_fe_values(slab->fluid_time_fe, quad, update_values | update_quadrature_points | update_JxW_values);
		std::vector<types::global_dof_index> time_local_dof_indices(slab->fluid_time_fe.n_dofs_per_cell());

		unsigned int n_local_snapshots = 0;

		for (const auto &time_cell : slab->fluid_time_dof_handler.active_cell_iterators())
		{
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				fluid_values_t_qq.push_back(t_qq);

				// get the FEM space solution at the quadrature point
				for (const unsigned int ii : time_fe_values.dof_indices())
				{
					for (unsigned int i = 0; i < fluid_n_space_dofs; ++i)
						solution_at_t_qq[n_local_snapshots](i) += solution(i + time_local_dof_indices[ii] * fluid_n_space_dofs + 0) * time_fe_values.shape_value(ii, qq);
				}

				n_local_snapshots++;

				// dG in time => repeat t_qq if num_splits > 1
				if (num_splits > 1 && qq > 0 && qq < quad.size()-1)
				{
					fluid_values_t_qq.push_back(t_qq);
					for (unsigned int i = 0; i < fluid_n_space_dofs; ++i)
						solution_at_t_qq[n_local_snapshots](i) = solution_at_t_qq[n_local_snapshots-1](i); 
					n_local_snapshots++;
				}
			}
		}
	}

	Assert(solution_at_t_qq.size() == fluid_values_t_qq.size(),
		ExcDimensionMismatch(solution_at_t_qq.size(), fluid_values_t_qq.size()));

	//////////
	// solid
	//
	{
		// Iterated formula splits temporal element into many subintervals
		// Should evaluate fluid and solid at same temporal points to compare in Paraview
		int num_splits = (initial_temporal_ref_fluid > initial_temporal_ref_solid) ? std::pow(2, initial_temporal_ref_fluid) : 1;
		QIterated<1> quad(QTrapez<1>(), num_splits);
		// std::cout << "quadrature formula size: " << quad.size() << std::endl;

		FEValues<1> time_fe_values(slab->solid_time_fe, quad, update_values | update_quadrature_points | update_JxW_values);
		std::vector<types::global_dof_index> time_local_dof_indices(slab->solid_time_fe.n_dofs_per_cell());

		unsigned int n_local_snapshots = 0;

		for (const auto &time_cell : slab->solid_time_dof_handler.active_cell_iterators()) {
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
			// time quadrature point
			double t_qq = time_fe_values.quadrature_point(qq)[0];
			solid_values_t_qq.push_back(t_qq);

			// get the FEM space solution at the quadrature point
			for (const unsigned int ii : time_fe_values.dof_indices())
			{
				for (unsigned int i = 0; i < solid_n_space_dofs; ++i)
					solution_at_t_qq[n_local_snapshots](i + fluid_n_space_dofs) += solution(i + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs) * time_fe_values.shape_value(ii, qq);
			}

			n_local_snapshots++;

			// dG in time => repeat t_qq if num_splits > 1
			if (num_splits > 1 && qq > 0 && qq < quad.size()-1)
			{
				solid_values_t_qq.push_back(t_qq);
				for (unsigned int i = 0; i < solid_n_space_dofs; ++i)
					solution_at_t_qq[n_local_snapshots](i + fluid_n_space_dofs) = solution_at_t_qq[n_local_snapshots-1](i + fluid_n_space_dofs); 
				n_local_snapshots++;
			}
			}
		}
	}

	Assert(solution_at_t_qq.size() == solid_values_t_qq.size(),
		ExcDimensionMismatch(solution_at_t_qq.size(), solid_values_t_qq.size()));

	for (unsigned int i = 0; i < solution_at_t_qq.size(); ++i)
		Assert(fluid_values_t_qq[i] == solid_values_t_qq[i], ExcNotImplemented());
}

template<>
void SpaceTime<1>::output_results(std::shared_ptr<Slab> &/*slab*/, const unsigned int refinement_cycle, unsigned int /*slab_number*/, bool last_slab) {
    if (!last_slab)
        return;

	std::string output_dir = "output/dim=1/cycle=" + std::to_string(refinement_cycle) + "/";
	std::string plotting_cmd = "python3 plot_solution.py  " + output_dir;
	system(plotting_cmd.c_str());
}

template<>
void SpaceTime<2>::output_results(
        std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number, bool last_slab) {
        std::string output_dir = "output/dim=2/cycle=" + std::to_string(refinement_cycle) + "/";

        // output results as VTK files
		std::vector<Vector<double>> solution_at_t_qq(
			std::max(slab->fluid_time_dof_handler.n_dofs(), slab->solid_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
			Vector<double>(space_dof_handler.n_dofs()) // joint solution at timepoint t_qq
		);
		std::vector<double> fluid_values_t_qq;
		std::vector<double> solid_values_t_qq;

		// fill the solutions on finer temporal mesh
		get_solution_on_finer_mesh(slab, solution_at_t_qq, fluid_values_t_qq, solid_values_t_qq);

		// for fluid and solid fill the solutions, at the end output the solution as vtk

		// //////////
		// // fluid
		// //
		// {
		// 	// Iterated formula splits temporal element into many subintervals
		// 	// Should evaluate fluid and solid at same temporal points to compare in Paraview
		// 	int num_splits = (initial_temporal_ref_fluid >= initial_temporal_ref_solid) ? 1 : std::pow(2, initial_temporal_ref_solid);
		// 	QIterated<1> quad(QTrapez<1>(), num_splits);
		// 	// std::cout << "quadrature formula size: " << quad.size() << std::endl;
		// 	FEValues<1> time_fe_values(slab->fluid_time_fe, quad, update_values | update_quadrature_points | update_JxW_values);
		// 	std::vector<types::global_dof_index> time_local_dof_indices(slab->fluid_time_fe.n_dofs_per_cell());

		// 	unsigned int n_local_snapshots = 0;

		// 	for (const auto &time_cell : slab->fluid_time_dof_handler.active_cell_iterators())
		// 	{
		// 		time_fe_values.reinit(time_cell);
		// 		time_cell->get_dof_indices(time_local_dof_indices);

		// 		for (const unsigned int qq : time_fe_values.quadrature_point_indices())
		// 		{
		// 			// time quadrature point
		// 			double t_qq = time_fe_values.quadrature_point(qq)[0];
		// 			fluid_values_t_qq.push_back(t_qq);

		// 			// get the FEM space solution at the quadrature point
		// 			for (const unsigned int ii : time_fe_values.dof_indices())
		// 			{
		// 				for (unsigned int i = 0; i < fluid_n_space_dofs; ++i)
		// 					solution_at_t_qq[n_local_snapshots](i) += solution(i + time_local_dof_indices[ii] * fluid_n_space_dofs + 0) * time_fe_values.shape_value(ii, qq);
		// 			}

		// 			n_local_snapshots++;

		// 			// dG in time => repeat t_qq if num_splits > 1
		// 			if (num_splits > 1 && qq > 0 && qq < quad.size()-1)
		// 			{
		// 				fluid_values_t_qq.push_back(t_qq);
		// 				for (unsigned int i = 0; i < fluid_n_space_dofs; ++i)
		// 					solution_at_t_qq[n_local_snapshots](i) = solution_at_t_qq[n_local_snapshots-1](i); 
		// 				n_local_snapshots++;
		// 			}
		// 		}
		// 	}
		// }

		// Assert(solution_at_t_qq.size() == fluid_values_t_qq.size(),
		//    ExcDimensionMismatch(solution_at_t_qq.size(), fluid_values_t_qq.size()));

        // //////////
        // // solid
        // //
        // {
        //     // Iterated formula splits temporal element into many subintervals
        //     // Should evaluate fluid and solid at same temporal points to compare in Paraview
        //     int num_splits = (initial_temporal_ref_fluid > initial_temporal_ref_solid) ? std::pow(2, initial_temporal_ref_fluid) : 1;
        //     QIterated<1> quad(QTrapez<1>(), num_splits);
        //     // std::cout << "quadrature formula size: " << quad.size() << std::endl;

        //     FEValues<1> time_fe_values(slab->solid_time_fe, quad, update_values | update_quadrature_points | update_JxW_values);
        //     std::vector<types::global_dof_index> time_local_dof_indices(slab->solid_time_fe.n_dofs_per_cell());

        //     unsigned int n_local_snapshots = 0;

        //     for (const auto &time_cell : slab->solid_time_dof_handler.active_cell_iterators()) {
        //       time_fe_values.reinit(time_cell);
        //       time_cell->get_dof_indices(time_local_dof_indices);

        //       for (const unsigned int qq : time_fe_values.quadrature_point_indices())
        //       {
        //         // time quadrature point
        //         double t_qq = time_fe_values.quadrature_point(qq)[0];
		// 		solid_values_t_qq.push_back(t_qq);

        //         // get the FEM space solution at the quadrature point
        //         for (const unsigned int ii : time_fe_values.dof_indices())
        //         {
        //           for (unsigned int i = 0; i < solid_n_space_dofs; ++i)
        //               solution_at_t_qq[n_local_snapshots](i + fluid_n_space_dofs) += solution(i + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs) * time_fe_values.shape_value(ii, qq);
        //         }

        //         n_local_snapshots++;

		// 		// dG in time => repeat t_qq if num_splits > 1
		// 		if (num_splits > 1 && qq > 0 && qq < quad.size()-1)
		// 		{
		// 			solid_values_t_qq.push_back(t_qq);
		// 			for (unsigned int i = 0; i < solid_n_space_dofs; ++i)
		// 				solution_at_t_qq[n_local_snapshots](i + fluid_n_space_dofs) = solution_at_t_qq[n_local_snapshots-1](i + fluid_n_space_dofs); 
		// 			n_local_snapshots++;
		// 		}
        //       }
        //     }
        // }

		// Assert(solution_at_t_qq.size() == solid_values_t_qq.size(),
		//    ExcDimensionMismatch(solution_at_t_qq.size(), solid_values_t_qq.size()));

		// for (unsigned int i = 0; i < solution_at_t_qq.size(); ++i)
		// 	Assert(fluid_values_t_qq[i] == solid_values_t_qq[i], ExcNotImplemented());

		// output the solution as vtk
		for (unsigned int i = 0; i < solution_at_t_qq.size(); i++)
		{
			DataOut<2> data_out;
			data_out.attach_dof_handler(space_dof_handler);

			std::vector<std::string> solution_names;
			solution_names.push_back("fluid_displacement");
			solution_names.push_back("fluid_velocity");
			solution_names.push_back("solid_displacement");
			solution_names.push_back("solid_velocity");

			std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(1+1+1+1, DataComponentInterpretation::component_is_scalar);
			// std::cout << "solution for i = " << i << std::endl;
			// solution_at_t_qq[i].print(std::cout);
			data_out.add_data_vector(solution_at_t_qq[i], solution_names, DataOut<2>::type_dof_data, data_component_interpretation);
			data_out.build_patches(1);

			data_out.set_flags(DataOutBase::VtkFlags(fluid_values_t_qq[i], n_snapshots));
			// std::cout << "t = " << fluid_values_t_qq[i] << " ;i = " << n_snapshots << std::endl;

			std::ofstream output(output_dir + "solution_" + Utilities::int_to_string(n_snapshots, 5) + ".vtk");
			data_out.write_vtk(output);

			n_snapshots++;
		}
}

template<int dim>
void SpaceTime<dim>::print_coordinates(std::shared_ptr<Slab> &slab,
		std::string output_dir, unsigned int slab_number) {
	////////////////////////
	// fluid
	//
	{
		// space
		std::vector<Point<dim>> space_support_points(space_dof_handler.n_dofs());
		DoFTools::map_dofs_to_support_points(
			MappingQ1<dim, dim>(),
			space_dof_handler,
			space_support_points
		);

		std::ofstream x_out(output_dir + "coordinates_fluid_x.txt");
		x_out.precision(9);
		x_out.setf(std::ios::scientific, std::ios::floatfield);

		unsigned int i = 0;
		for (auto point : space_support_points) {
			if (i == fluid_n_space_u) // only output coordinates for u_f
				break;
			x_out << point[0] << ' ';
			i++;
		}

		// time
		std::vector<Point<1>> time_support_points(slab->fluid_time_dof_handler.n_dofs());
		DoFTools::map_dofs_to_support_points(
			MappingQ1<1, 1>(),
			slab->fluid_time_dof_handler,
			time_support_points
		);

		std::ofstream t_out(output_dir + "coordinates_fluid_t_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		t_out.precision(9);
		t_out.setf(std::ios::scientific, std::ios::floatfield);

		for (auto point : time_support_points)
			t_out << point[0] << ' ';
		t_out << std::endl;
	}

	////////////////////////
	// solid
	//
	{
		// space
		std::vector<Point<dim>> space_support_points(space_dof_handler.n_dofs());
		DoFTools::map_dofs_to_support_points(
			MappingQ1<dim, dim>(),
			space_dof_handler,
			space_support_points
		);

		std::ofstream x_out(output_dir + "coordinates_solid_x.txt");
		x_out.precision(9);
		x_out.setf(std::ios::scientific, std::ios::floatfield);

		unsigned int i = 0;
		for (auto point : space_support_points) {
			if (i < fluid_n_space_dofs) // skip fluid coordinates
			{
				++i;
				continue;
			}
			if (i == fluid_n_space_dofs + solid_n_space_u) // only output coordinates for u_s
				break;

			x_out << point[0] << ' ';
			i++;
		}

		// time
		std::vector<Point<1>> time_support_points(slab->solid_time_dof_handler.n_dofs());
		DoFTools::map_dofs_to_support_points(
			MappingQ1<1, 1>(),
			slab->solid_time_dof_handler,
			time_support_points
		);

		std::ofstream t_out(output_dir + "coordinates_solid_t_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		t_out.precision(9);
		t_out.setf(std::ios::scientific, std::ios::floatfield);

		for (auto point : time_support_points)
			t_out << point[0] << ' ';
		t_out << std::endl;
	}

}

template<int dim>
void SpaceTime<dim>::print_solution(std::shared_ptr<Slab> &slab,
		std::string output_dir, unsigned int slab_number) {
	//////////
	// fluid
	{
		Vector<double> solution_u(fluid_n_space_u * slab->fluid_time_dof_handler.n_dofs());
		Vector<double> solution_v(fluid_n_space_v * slab->fluid_time_dof_handler.n_dofs());
		unsigned int index_u = 0;
		unsigned int index_v = 0;
		for (unsigned int ii = 0; ii < slab->fluid_time_dof_handler.n_dofs(); ++ii) {
			// displacement
			for (unsigned int j = ii * fluid_n_space_dofs; j < ii * fluid_n_space_dofs + fluid_n_space_u; ++j) {
				solution_u[index_u] = solution[j];
				index_u++;
			}
			// velocity
			for (unsigned int j = ii * fluid_n_space_dofs + fluid_n_space_u; j < (ii + 1) * fluid_n_space_dofs; ++j) {
				solution_v[index_v] = solution[j];
				index_v++;
			}
		}
		std::ofstream solution_u_out(output_dir + "solution_fluid_u_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		solution_u.print(solution_u_out, /*precision*/16);
		std::ofstream solution_v_out(output_dir + "solution_fluid_v_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		solution_v.print(solution_v_out, /*precision*/16);
	}

	//////////
	// solid
	{
		Vector<double> solution_u(solid_n_space_u * slab->solid_time_dof_handler.n_dofs());
		Vector<double> solution_v(solid_n_space_v * slab->solid_time_dof_handler.n_dofs());
		unsigned int index_u = 0;
		unsigned int index_v = 0;
		for (unsigned int ii = 0; ii < slab->solid_time_dof_handler.n_dofs(); ++ii) {
			// displacement
			for (unsigned int j = ii * solid_n_space_dofs; j < ii * solid_n_space_dofs + solid_n_space_u; ++j) {
				solution_u[index_u] = solution[j + fluid_n_dofs];
				index_u++;
			}
			// velocity
			for (unsigned int j = ii * solid_n_space_dofs + solid_n_space_u; j < (ii + 1) * solid_n_space_dofs; ++j) {
				solution_v[index_v] = solution[j + fluid_n_dofs];
				index_v++;
			}
		}
		std::ofstream solution_u_out(output_dir + "solution_solid_u_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		solution_u.print(solution_u_out, /*precision*/16);
		std::ofstream solution_v_out(output_dir + "solution_solid_v_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		solution_v.print(solution_v_out, /*precision*/16);
	}
}

template<int dim>
void SpaceTime<dim>::print_error(std::shared_ptr<Slab> &slab,
		std::string output_dir, unsigned int slab_number) {
	
	//////////
	// fluid
	{
		Solution<dim> solution_func;
		FEValues<1> time_fe_values(slab->fluid_time_fe, Quadrature<1>(slab->fluid_time_fe.get_unit_support_points()), update_quadrature_points);
		const unsigned int time_dofs_per_cell = slab->fluid_time_fe.n_dofs_per_cell();
		auto time_cell = slab->fluid_time_dof_handler.begin_active();

		Vector<double> error_u(fluid_n_space_u * slab->fluid_time_dof_handler.n_dofs());
		Vector<double> error_v(fluid_n_space_v * slab->fluid_time_dof_handler.n_dofs());
		unsigned int index_u = 0;
		unsigned int index_v = 0;
		for (unsigned int ii = 0; ii < slab->fluid_time_dof_handler.n_dofs(); ++ii) {
			// reinit temporal FEValues
			if (ii % time_dofs_per_cell == 0)
			{
				// go to next temporal_cell
				if (ii > 0)
					time_cell++;

				time_fe_values.reinit(time_cell);
			}
			
			// get the analytical space solution at the quadrature point
			double t_qq = time_fe_values.quadrature_point(ii % time_dofs_per_cell)[0];
			solution_func.set_time(t_qq);
			Vector<double> analytical_solution(space_dof_handler.n_dofs());
			VectorTools::interpolate(space_dof_handler,
						solution_func,
						analytical_solution,
						ComponentMask());
			
			// displacement
			for (unsigned int j = ii * fluid_n_space_dofs; j < ii * fluid_n_space_dofs + fluid_n_space_u; ++j) {
				error_u[index_u] = solution[j] - analytical_solution[j - ii*fluid_n_space_dofs];
				index_u++;
			}
			// velocity
			for (unsigned int j = ii * fluid_n_space_dofs + fluid_n_space_u; j < (ii + 1) * fluid_n_space_dofs; ++j) {
				error_v[index_v] = solution[j] - analytical_solution[j - ii*fluid_n_space_dofs];
				index_v++;
			}
		}
		std::ofstream error_u_out(output_dir + "error_fluid_u_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		error_u.print(error_u_out, /*precision*/16);
		std::ofstream error_v_out(output_dir + "error_fluid_v_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		error_v.print(error_v_out, /*precision*/16);
	}

	//////////
	// solid
	{
		Solution<dim> solution_func;
		FEValues<1> time_fe_values(slab->solid_time_fe, Quadrature<1>(slab->solid_time_fe.get_unit_support_points()), update_quadrature_points);
		const unsigned int time_dofs_per_cell = slab->solid_time_fe.n_dofs_per_cell();
		auto time_cell = slab->solid_time_dof_handler.begin_active();

		Vector<double> error_u(solid_n_space_u * slab->solid_time_dof_handler.n_dofs());
		Vector<double> error_v(solid_n_space_v * slab->solid_time_dof_handler.n_dofs());
		unsigned int index_u = 0;
		unsigned int index_v = 0;
		for (unsigned int ii = 0; ii < slab->solid_time_dof_handler.n_dofs(); ++ii) {
			// reinit temporal FEValues
			if (ii % time_dofs_per_cell == 0)
			{
				// go to next temporal_cell
				if (ii > 0)
					time_cell++;

				time_fe_values.reinit(time_cell);
			}
			
			// get the analytical space solution at the quadrature point
			double t_qq = time_fe_values.quadrature_point(ii % time_dofs_per_cell)[0];
			solution_func.set_time(t_qq);
			Vector<double> analytical_solution(space_dof_handler.n_dofs());
			VectorTools::interpolate(space_dof_handler,
						solution_func,
						analytical_solution,
						ComponentMask());

			// displacement
			for (unsigned int j = ii * solid_n_space_dofs; j < ii * solid_n_space_dofs + solid_n_space_u; ++j) {
				error_u[index_u] = solution[j + fluid_n_dofs] - analytical_solution[j - ii*solid_n_space_dofs + fluid_n_space_dofs];
				index_u++;
			}
			// velocity
			for (unsigned int j = ii * solid_n_space_dofs + solid_n_space_u; j < (ii + 1) * solid_n_space_dofs; ++j) {
				error_v[index_v] = solution[j + fluid_n_dofs] - analytical_solution[j - ii*solid_n_space_dofs + fluid_n_space_dofs];
				index_v++;
			}
		}
		std::ofstream error_u_out(output_dir + "error_solid_u_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		error_u.print(error_u_out, /*precision*/16);
		std::ofstream error_v_out(output_dir + "error_solid_v_" + Utilities::int_to_string(slab_number, 5) + ".txt");
		error_v.print(error_v_out, /*precision*/16);
	}
}

template<int dim>
void SpaceTime<dim>::run() {
	// create a coarse grid
	make_grids();

	// Refinement loop
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle) {
		std::cout
				<< "-------------------------------------------------------------"
				<< std::endl;
		std::cout << "|                REFINEMENT CYCLE: " << cycle;
		std::cout << "                         |" << std::endl;
		std::cout
				<< "-------------------------------------------------------------"
				<< std::endl;

		// reset values from last refinement cycle
		fluid_total_n_dofs = 0;
		solid_total_n_dofs = 0;
		L2_error = 0.;
		L2_fluid_error = 0.;
		L2_solid_error = 0.;
		goal_func_value = 0.;
		n_snapshots = 0;

		// ComponentMask fluid_mask(4, false);
		// fluid_mask.set(0, true);
		// fluid_mask.set(1, true);
		// ComponentMask solid_mask(4, false);
		// solid_mask.set(2, true);
		// solid_mask.set(3, true);

		// create output directory if necessary
        std::string dim_dir = "output/dim=" + std::to_string(dim) + "/";
        std::string output_dir = dim_dir + "cycle=" + std::to_string(cycle) + "/";
        for (auto dir : { "output/", dim_dir.c_str(), output_dir.c_str() })
            mkdir(dir, S_IRWXU);

		////////////////////////////////////////////
		// create spatial DoFHandler
		//
		set_active_fe_indices();
		space_dof_handler.distribute_dofs(space_fe_collection);
		
		// Renumber spatials DoFs into fluid_displacement, fluid_velocity, solid_displacement and solid_velocity DoFs
  		DoFRenumbering::component_wise(space_dof_handler, {0, 1, 2, 3});

		// four blocks: fluid_displacement, fluid_velocity, solid_displacement and solid_velocity
		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(space_dof_handler, {0, 1, 2, 3});
		fluid_n_space_u = dofs_per_block[0];
		fluid_n_space_v = dofs_per_block[1];
		solid_n_space_u = dofs_per_block[2];
		solid_n_space_v = dofs_per_block[3];

		// number of space DoFs per domain
		fluid_n_space_dofs = fluid_n_space_u + fluid_n_space_v;
		solid_n_space_dofs = solid_n_space_u + solid_n_space_v;

		// std::cout << "Renumbered DoFs: " << std::endl;

		////////////////////////////////////////////
		// initial value: u(0) and v(0)
		//

		// compute fluid and solid initial value vector
		Vector<double> initial_solution(space_dof_handler.n_dofs());
		VectorTools::interpolate(
			space_dof_handler,
			InitialValues<dim>(),
			initial_solution,
			ComponentMask()
		);

		// fill initial solution for fluid and solid
		initial_solution_fluid.reinit(space_dof_handler.n_dofs());
		initial_solution_solid.reinit(space_dof_handler.n_dofs());
		// fluid
		for (unsigned int i = 0; i < fluid_n_space_dofs; ++i)
			initial_solution_fluid(i) = initial_solution(i);
		// solid
		for (unsigned int i = 0; i < solid_n_space_dofs; ++i)
			initial_solution_solid(i + fluid_n_space_dofs) = initial_solution(i + fluid_n_space_dofs);

		/////////////////////////////
		//    TIME-SLABBING LOOP
		//
		for (unsigned int k = 0; k < slabs.size(); ++k) {
			// create and solve linear system
			setup_system(slabs[k], k+1);		
			assemble_system(slabs[k], k==0);
			solve(k == 0);

			// write system matrix out to file on the first slab to compute the condition number
			if (k == 0)
			{
				std::ofstream matrix_out(output_dir + "matrix.txt");
				//print_as_numpy_arrays_high_resolution(system_matrix, matrix_out, /*precision*/16);
			}

#if DIM == 1
			// output Space-Time DoF coordinates
			print_coordinates(slabs[k], output_dir, k);

			// output solution to txt file
			print_solution(slabs[k], output_dir, k);

			// output error of FEM solution to txt file
			print_error(slabs[k], output_dir, k);

			// output vtk OR run python script to create solution contour plot
			output_results(slabs[k], cycle, k, (k == slabs.size()-1));
#endif
			// Compute the error to the analytical solution
			process_solution(slabs[k], cycle, (k == slabs.size()-1));

#if DIM == 2
			// compute goal functional
			compute_goal_functional(slabs[k]);
#endif

			///////////////////////
			// prepare next slab
			//

			// NOTE: this getting of initial values only works for QGaussLobatto in time, when the last temporal index corresponds to the last temporal quadrature point on the temporal element 
			// get initial value for next slab
			// fluid
			for (unsigned int i = 0; i < fluid_n_space_dofs; ++i)
				initial_solution_fluid(i) = solution(i + fluid_n_dofs - fluid_n_space_dofs);
			// solid
			for (unsigned int i = 0; i < solid_n_space_dofs; ++i)
				initial_solution_solid(i + fluid_n_space_dofs) = solution(i + solid_n_dofs - solid_n_space_dofs + fluid_n_dofs);
		}

		goal_func_vals.push_back(goal_func_value);

		// refine mesh
		if (cycle < max_n_refinement_cycles - 1) {
			space_triangulation.refine_global(refine_space);
			
			if (split_slabs) {
				std::vector<std::shared_ptr<Slab> > split_slabs;
				for (auto &slab : slabs) {
					// NOTE: using same temporal degree for fluid and solid
					Assert(slab->fluid_time_fe.get_degree() == slab->solid_time_fe.get_degree(), ExcNotImplemented());
					split_slabs.push_back(
						std::make_shared<Slab>(
							slab->fluid_time_fe.get_degree(),
							slab->start_time,
							0.5 * (slab->start_time + slab->end_time)
						)
					);
					split_slabs.push_back(
						std::make_shared<Slab>(
							slab->fluid_time_fe.get_degree(),
							0.5 * (slab->start_time + slab->end_time),
							slab->end_time
						)
					);
				}
				slabs = split_slabs;

				for (auto &slab : slabs) {
					GridGenerator::hyper_rectangle(
						slab->fluid_time_triangulation,
						Point<1>(slab->start_time),
						Point<1>(slab->end_time)
					);
					slab->fluid_time_triangulation.refine_global(initial_temporal_ref_fluid);
					GridGenerator::hyper_rectangle(
						slab->solid_time_triangulation,
						Point<1>(slab->start_time),
						Point<1>(slab->end_time)
					);
					slab->solid_time_triangulation.refine_global(initial_temporal_ref_solid);
				}
			} else {
				for (auto &slab : slabs) {
					slab->fluid_time_triangulation.refine_global(refine_time);
					slab->solid_time_triangulation.refine_global(refine_time);
				}
			}
		}
	}

	print_convergence_table();

#if DIM == 2
	std::cout << "Goal functional values:" << std::endl;
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle) {
		std::cout.precision(15);
		std::cout.setf(std::ios::scientific, std::ios::floatfield);
		std::cout << "Cycle " << cycle << ": " << goal_func_vals[cycle] << std::endl;
	}

	double reference_value = 0.;
	if (!SOLID_SOURCE)
		reference_value = 2.485876918504151e-04; // N = 50,000; fluid_ref = 0; solid_ref = 0 // fluid source term
	else
		reference_value = 7.142768240616408e-04; // N = 50,000; fluid_ref = 0; solid_ref = 0 // solid source term
	std::cout << "Error in goal functional values:" << std::endl;
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle) {
		std::cout.precision(15);
		std::cout.setf(std::ios::scientific, std::ios::floatfield);
		std::cout << "Cycle " << cycle << ": " << std::abs(goal_func_vals[cycle] - reference_value) << std::endl;
	}
#endif


#if CONDITION
	// print condition numbers
	std::cout << "\nCondition numbers: " << std::endl;
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle) {
		std::string output_dir = "output/dim=" + std::to_string(dim) + "/cycle=" + std::to_string(cycle) + "/";
		std::string condition_cmd = "python3 condition_number.py  " + output_dir;
		system(condition_cmd.c_str());
	}
#endif
}

int main(int argc, char** argv) {
	try {
		deallog.depth_console(2);

		double gamma = 1000.;
		// use finer temporal fluid grid as default
		unsigned int solid_ref = 0;
		unsigned int fluid_ref = 1;
		// NOTE: ref = (0,4) or ref = (4,0) would already result in 16 : 1 temporal elements
		//       this can become expensive very quickly, since in 2+1D and 80x20 spatial cells we have 60,000 Space-Time DoFs

		// parse command line arguments
		if (argc > 1)
		{
			for (int i = 1; i < argc; ++i)
			{
				if (std::string(argv[i]) == std::string("-gamma"))
				{
					gamma = std::stod(argv[i+1]);
					i++;
				}
				else if (std::string(argv[i]) == std::string("-solid_ref"))
				{
					solid_ref = std::stoi(argv[i+1]);
					i++;
				}
				else if (std::string(argv[i]) == std::string("-fluid_ref"))
				{
					fluid_ref = std::stoi(argv[i+1]);
					i++;
				}
			}
		}
		std::cout << "CLI ARGUMENTS:" << std::endl;
		std::cout << "fluid_ref = " << fluid_ref << std::endl;
		std::cout << "solid_ref = " << solid_ref << std::endl;
		std::cout << "gamma     = " << gamma << std::endl << std::endl;

		// run the simulation
#if DIM == 1
		// 1+1D:
		// -----
		SpaceTime<1> space_time_problem(
				1,                      // s ->  spatial FE degree
				{ 1, 1, 1, 1 },         // r -> temporal FE degree
				{ 0., 1., 2., 3., 4. }, // time points
				6, //4,//5,                      // max_n_refinement_cycles,
				fluid_ref,              // initial_temporal_ref_fluid
				solid_ref,              // initial_temporal_ref_solid
				true,                   // refine_space
				true,                   // refine_time
				true,                   // split_slabs
				gamma                   // gamma - penalty parameter
		);
#elif DIM == 2
		// 2+1D:
		// -----
		std::vector<unsigned int> r;
		std::vector<double> t = { 0. };
		double T = 1.;
		int N = 50; // 50000;
		double dt = T / N;
		for (unsigned int i = 0; i < N; ++i) {
			r.push_back(1); 
			t.push_back((i + 1) * dt);
		} 
        SpaceTime<2> space_time_problem(
			1,                      // s ->  spatial FE degree
			r,         				// r -> temporal FE degree
			t, 						// time points
			4,                      // max_n_refinement_cycles,
			fluid_ref,              // initial_temporal_ref_fluid
			solid_ref,              // initial_temporal_ref_solid
			false,                  // refine_space
			true,                   // refine_time
			true,                   // split_slabs
			gamma                   // gamma - penalty parameter
		);
#endif

		// run the simulation
		space_time_problem.run();

		// save final grid
		//space_time_problem.print_grids(
		//	"space_grid.svg", 
		//	"time_grid_fluid.svg", 
		//	"time_grid_solid.svg", 
		//	"time_grid_joint.svg"
		//);
	} catch (std::exception &exc) {
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl << exc.what()
				<< std::endl << "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	} catch (...) {
		std::cerr << std::endl << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl << "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
