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
 * TENSOR-PRODUCT SPACE-TIME FINITE ELEMENTS:
 * ==========================================
 * Tensor-product space-time code for the Mandel problem with Q^s finite elements in space and dG-Q^r finite elements in time: cG(s)cG(r)
 * We use multirate finite elements in time, where e.g. for each temporal element of the pressure we have four temporal elements of the displacement.
 * 
 * Author: Julian Roth, 2023
 * 
 * For more information on tensor-product space-time finite elements please also check out the DTM-project by Uwe Köcher and contributors.
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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

// output vtk
#ifndef VTK_OUTPUT
#  define VTK_OUTPUT false
#endif

// For Mandel's problem we have DIM = 2
#ifndef DIM
#  define DIM 2
#endif

using namespace dealii;

////////////////////////////////////////
// parameters (provided by Thomas Wick)
//

// M_biot = Biot's constant
// double M_biot = 2.5e+12; 
double M_biot = 1.75e+7; // value from Liu
double c_biot = 1.0/M_biot;

// alpha_biot = b_biot = Biot's modulo
double alpha_biot =  1.0; 
double viscosity_biot = 1.0e-3; 
double K_biot = 1.0e-13; 
double density_biot = 1.0;

// Traction 
double traction_x_biot = 0.0;
double traction_y_biot = -1.0e+7;

// Solid parameters
double density_structure = 1.0; 
double lame_coefficient_mu = 1.0e+8; 
double poisson_ratio_nu = 0.2; 
double lame_coefficient_lambda =  (2 * poisson_ratio_nu * lame_coefficient_mu) / (1.0 - 2 * poisson_ratio_nu);


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
		Function<dim>(dim+1) {
	}

	virtual double value(const Point<dim> &p,
			const unsigned int component) const override;

	virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;
};

template<int dim>
double InitialValues<dim>::value(const Point<dim> &/*p*/,
		const unsigned int /*component*/) const {
	// NOTE: According to GIRAULT, PENCHEVA, WHEELER and WILDEY 2011
    //       the initial conditions are p = 0, u_x = 0, u_y = 0., cf. page 210.
	return 0.;
}

template <int dim>
void InitialValues<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<dim+1; ++c)
    values(c) = InitialValues<dim>::value(p, c);
}

template<int dim>
class BoundaryValues: public Function<dim> {
public:
	BoundaryValues() : 
		Function<dim>(dim+1) {
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
  for (unsigned int c=0; c<dim+1; ++c)
    values(c) = BoundaryValues<dim>::value(p, c);
}

template <int dim>
class Solution: public Function<dim> {
public:
    Solution() : Function<dim>(dim+1) {}

	virtual	double value(const Point<dim> &p,
	        const unsigned int component) const override;
		
	virtual void vector_value (const Point<dim> &p,
	        Vector<double>   &value) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim> &/*p*/,
		const unsigned int /*component*/) const {
	//const double t = this->get_time();
	return 0.; // no analytical solution is known
}

template <int dim>
void Solution<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const
{
  for (unsigned int c=0; c<dim+1; ++c)
    values(c) = Solution<dim>::value(p, c);
}

class Slab {
public:
	// constructor
	Slab(unsigned int r, double start_time, double end_time);

	////////////////
	// displacement
	//
	// time
	Triangulation<1> displacement_time_triangulation;
	FE_DGQ<1> displacement_time_fe;
	DoFHandler<1> displacement_time_dof_handler;

	////////////
	// pressure
	//
	// time
	Triangulation<1> pressure_time_triangulation;
	FE_DGQ<1> pressure_time_fe;
	DoFHandler<1> pressure_time_dof_handler;

	double start_time, end_time;
};

// NOTE: Use QGaussLobatto quadrature in time. For QGauss one would need to change how the initial value is computed for the next slab in run().
//       Moreover, the temporal interpolation matrix would need to be adapted.
Slab::Slab(unsigned int r, double start_time, double end_time) :
		displacement_time_fe(r), displacement_time_dof_handler(displacement_time_triangulation), pressure_time_fe(
				r), pressure_time_dof_handler(pressure_time_triangulation), start_time(
				start_time), end_time(end_time) {
}

template<int dim>
class SpaceTime {
public:
	SpaceTime(int s_displacement,
			int s_pressure, 
			std::vector<unsigned int> r,
			std::vector<double> time_points = { 0., 1. },
			unsigned int max_n_refinement_cycles = 3,
			unsigned int initial_temporal_ref_displacement = 0,
			unsigned int initial_temporal_ref_pressure = 1,
			bool refine_space = true,
			bool refine_time = true,
			bool split_slabs = true);
	void run();
	void print_grids(std::string file_name_space, std::string file_name_time_displacement, std::string file_name_time_pressure, std::string file_name_time_joint);
	void print_convergence_table();

private:
	void make_grids();
	void setup_system(std::shared_ptr<Slab> &slab, unsigned int k);
	void assemble_system(std::shared_ptr<Slab> &slab, bool assemble_matrix);
	void apply_boundary_conditions(std::shared_ptr<Slab> &slab);
	void solve(bool invert);
	void get_solution_on_finer_mesh(std::shared_ptr<Slab> &slab, std::vector<Vector<double>> &solution_at_t_qq, std::vector<double> &displacement_values_t_qq, std::vector<double> &pressure_values_t_qq);
	void output_results(std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number, bool last_slab);
	void process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab);
	void compute_goal_functional(std::shared_ptr<Slab> &slab);
	void compute_functional_values(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool first_slab);

	//////////
	// space
	//
	Triangulation<dim>    space_triangulation;
	FESystem<dim>         space_fe;
  	DoFHandler<dim>       space_dof_handler;
	Vector<double> 		  initial_solution_displacement;
	Vector<double> 		  initial_solution_pressure;
	
	//////////
	// time
	//
	std::vector<std::shared_ptr<Slab> > slabs;

	// displacement
	std::set< std::pair<double, unsigned int> > displacement_time_support_points; // (time_support_point, support_point_index)

	types::global_dof_index       displacement_n_space_dofs;
	types::global_dof_index       displacement_n_dofs; // space-time DoFs
		
	// pressure
	std::set< std::pair<double, unsigned int> > pressure_time_support_points; // (time_support_point, support_point_index)

	types::global_dof_index       pressure_n_space_dofs;
	types::global_dof_index       pressure_n_dofs; // space-time DoFs

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

	const FEValuesExtractors::Vector displacement = 0;
  	const FEValuesExtractors::Scalar pressure     = dim;

	// maps from coordinate to DoF index for pressure and x-displacement on bottom
	// boundary
	std::map<double, types::global_dof_index> pressure_dofmap;
	std::map<double, types::global_dof_index> x_disp_dofmap;

	double start_time, end_time;
	
	unsigned int n_snapshots;
	unsigned int max_n_refinement_cycles;
	unsigned int initial_temporal_ref_displacement;
	unsigned int initial_temporal_ref_pressure;
	bool refine_space, refine_time, split_slabs;

	double L2_error;
	double goal_func_value;
	std::vector<double> goal_func_vals;
	std::vector<double> L2_error_vals;
	unsigned int displacement_total_n_dofs;
	unsigned int pressure_total_n_dofs;
	ConvergenceTable convergence_table;
};

template <int dim>
SpaceTime<dim>::SpaceTime(
	int s_displacement,
	int s_pressure, 
	std::vector<unsigned int> r, 
	std::vector<double> time_points,
	unsigned int max_n_refinement_cycles,
	unsigned int initial_temporal_ref_displacement,
	unsigned int initial_temporal_ref_pressure,
	bool refine_space,
	bool refine_time,
	bool split_slabs) : 
		space_fe(
			/*u*/ FE_Q<dim>(s_displacement), dim,
			/*p*/ FE_Q<dim>(s_pressure), 1),
		space_dof_handler(space_triangulation),

		max_n_refinement_cycles(max_n_refinement_cycles),
		initial_temporal_ref_displacement(initial_temporal_ref_displacement),
		initial_temporal_ref_pressure(initial_temporal_ref_pressure),
		refine_space(refine_space),
		refine_time(refine_time),
		split_slabs(split_slabs)
{
	// time_points = [t_0, t_1, ..., t_M]
	// r = [r_1, r_2, ..., r_M] with r_k is the temporal FE degree on I_k = (t_{k-1},t_k]
	Assert(r.size() + 1 == time_points.size(),
		   ExcDimensionMismatch(r.size() + 1, time_points.size()));
	// NOTE: at first hard coding r = 0 as the temporal degree
	for (unsigned int k = 0; k < r.size(); ++k)
		Assert(r[k] == 0, ExcNotImplemented());

	// create slabs
	for (unsigned int k = 0; k < r.size(); ++k)
	{
		slabs.push_back(
			std::make_shared<Slab>(r[k], time_points[k],
								   time_points[k + 1]));
	}

	start_time = time_points[0];
	end_time = time_points[time_points.size() - 1];
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
		bool displacement) {
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
		auto time_triangulation = ((displacement) ?  &(slab->displacement_time_triangulation) : &(slab->pressure_time_triangulation));
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
	// print joint displacement/pressure temporal triangulation, e.g. displacement above timeline and pressure below timeline
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
		for (auto time_triangulation : {&(slab->pressure_time_triangulation), &(slab->displacement_time_triangulation)})
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
void SpaceTime<2>::print_grids(std::string file_name_space, std::string file_name_time_displacement, std::string file_name_time_pressure, std::string file_name_time_joint) {

	//////////
	// space
	//
	
	std::ofstream out_space(file_name_space);
	GridOut grid_out_space;

	GridOutFlags::Svg svg_flags;
	svg_flags.label_boundary_id              = true;

	grid_out_space.set_flags(svg_flags);
	grid_out_space.write_svg(space_triangulation, out_space);

	//////////
	// time
	//

	// displacement
	std::ofstream out_time_displacement(file_name_time_displacement);
	print_1d_grid_slabwise(out_time_displacement, slabs, start_time, end_time, true);

	// pressure
	std::ofstream out_time_pressure(file_name_time_pressure);
	print_1d_grid_slabwise(out_time_pressure, slabs, start_time, end_time, false);

	// joint: displacement + pressure
	std::ofstream out_time_joint(file_name_time_joint);
	print_1d_grid_slabwise_joint(out_time_joint, slabs, start_time, end_time);
}

template<>
void SpaceTime<2>::make_grids() {
	//////////
	// space
	//
	std::string grid_name;
	grid_name  = "rectangle_mandel.inp"; 
	
	GridIn<2> grid_in;
	grid_in.attach_triangulation(space_triangulation);
	std::ifstream input_file(grid_name.c_str());      
	grid_in.read_ucd(input_file); 
		
	space_triangulation.refine_global(4); 

	// // FOR DEBUGGING: create coarse spatial mesh
	// GridGenerator::hyper_rectangle(space_triangulation, Point<2>(0., 0.),
	// 								Point<2>(100., 20.));
	// // space_triangulation.refine_global(1);

	// for (auto &cell : space_triangulation.cell_iterators())
	// 	for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell;
	// 	face++)
	// 	if (cell->face(face)->at_boundary()) {
	// 		if (cell->face(face)->center()[0] < 0. + 1e-10)
	// 		cell->face(face)->set_boundary_id(0);
	// 		else if (cell->face(face)->center()[0] > 100. - 1e-10)
	// 		cell->face(face)->set_boundary_id(1);
	// 		else if (cell->face(face)->center()[1] < 0. + 1e-10)
	// 		cell->face(face)->set_boundary_id(2);
	// 		else if (cell->face(face)->center()[1] > 20. - 1e-10)
	// 		cell->face(face)->set_boundary_id(3);
	// 	}
	
	//////////
	// time
    //

    // displacement
	for (auto &slab: slabs)
	{
		// create temporal displacement grid on slab
	    GridGenerator::hyper_rectangle(
			slab->displacement_time_triangulation,
			Point<1>(slab->start_time),
			Point<1>(slab->end_time)
		);
		// globally refine the temporal grid
		slab->displacement_time_triangulation.refine_global(initial_temporal_ref_displacement);
	}

	// pressure
	for (auto &slab: slabs)
	{
		// create temporal pressure grid on slab
	    GridGenerator::hyper_rectangle(
			slab->pressure_time_triangulation,
			Point<1>(slab->start_time),
			Point<1>(slab->end_time)
		);
		// globally refine the temporal grid
		slab->pressure_time_triangulation.refine_global(initial_temporal_ref_pressure);
	}
}

template<int dim>
void SpaceTime<dim>::setup_system(std::shared_ptr<Slab> &slab, unsigned int k) {
	///////////////////
	// distribute DoFs

	// time	
	slab->displacement_time_dof_handler.distribute_dofs(slab->displacement_time_fe);
	slab->pressure_time_dof_handler.distribute_dofs(slab->pressure_time_fe);

	// number of space-time DoFs per domain
	displacement_n_dofs = displacement_n_space_dofs * slab->displacement_time_dof_handler.n_dofs();
	pressure_n_dofs = pressure_n_space_dofs * slab->pressure_time_dof_handler.n_dofs();

	std::cout.setf(std::ios::scientific, std::ios::floatfield);
	std::cout << "Slab Q_" << k << " = Ω x (" << slab->start_time << "," << slab->end_time << "):" << std::endl;
	std::cout << "==============================================" << std::endl;
	std::cout << "#DoFs:" << std::endl << "------" << std::endl;
	std::cout << "displacement: " <<  displacement_n_space_dofs << " (space), "
		  	  << slab->displacement_time_dof_handler.n_dofs() << " (time)" << std::endl;
  	std::cout << "pressure: " <<  pressure_n_space_dofs << " (space), "
		 	  << slab->pressure_time_dof_handler.n_dofs() << " (time)" << std::endl;
	std::cout << "Total: " <<  displacement_n_dofs + pressure_n_dofs 
	          << " (space-time)" << std::endl  << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////////
	// space-time sparsity pattern = tensor product of spatial and temporal sparsity pattern
	//
	
	// linear problem with constant temporal and spatial meshes: sparsity pattern remains the same on future slabs 
	if (k == 1) {
		// NOTE: For simplicity, we assume that either the displacement or the pressure temporal triangulation contains only one element
		//                      OR both temporal triangulations have the same number of temporal elements, since then the temporal interpolation matrix is the identity matrix
		Assert(initial_temporal_ref_displacement == 0 || initial_temporal_ref_pressure == 0 || initial_temporal_ref_displacement == initial_temporal_ref_pressure, ExcNotImplemented());
		// NOTE: This one temporal element then has 1 temporal DoFs for dG(0) OR both time dof handlers have the same number of DoFs
		Assert(slab->displacement_time_dof_handler.n_dofs() == 1 || slab->pressure_time_dof_handler.n_dofs() == 1 
		|| slab->displacement_time_dof_handler.n_dofs() == slab->pressure_time_dof_handler.n_dofs(),
			ExcNotImplemented());

		/////////////////////////////////
		// temporal interpolation matrix
		double larger_time_n_dofs = std::max(
			slab->displacement_time_dof_handler.n_dofs(),
			slab->pressure_time_dof_handler.n_dofs()
		);
		double smaller_time_n_dofs = std::min(
			slab->displacement_time_dof_handler.n_dofs(),
			slab->pressure_time_dof_handler.n_dofs()
		);
		double larger_time_n_cells = std::max(
			slab->displacement_time_triangulation.n_active_cells(),
			slab->pressure_time_triangulation.n_active_cells()
		);

		DynamicSparsityPattern temp_interp_dsp(larger_time_n_dofs, smaller_time_n_dofs);

		if (larger_time_n_dofs == smaller_time_n_dofs)
		{
			// NOTE: temporal interpolation matrix is just the identity matrix
			
			// create sparsit pattern for temporal interpolation matrix by hand
			for (unsigned int i = 0; i < larger_time_n_dofs; ++i)
				temp_interp_dsp.add(i, i);
			
			temporal_interpolation_sparsity_pattern.copy_from(temp_interp_dsp);
			temporal_interpolation_matrix.reinit(temporal_interpolation_sparsity_pattern);

			// fill temporal_interpolation_matrix
			for (unsigned int i = 0; i < larger_time_n_dofs; ++i)
				temporal_interpolation_matrix.add(i, i, 1.);
		}
		else
		{
			Assert(smaller_time_n_dofs == 1, ExcNotImplemented());

			// create sparsity pattern for temporal interpolation matrix by hand
			for (unsigned int i = 0; i < larger_time_n_dofs; ++i) {
				temp_interp_dsp.add(i, 0);
			}

			temporal_interpolation_sparsity_pattern.copy_from(temp_interp_dsp);
			temporal_interpolation_matrix.reinit(temporal_interpolation_sparsity_pattern);

			// fill temporal_interpolation_matrix
			for (unsigned int i = 0; i < larger_time_n_dofs; ++i) {
				temporal_interpolation_matrix.add(i, 0, 1.0);
			}
		}

#ifdef DEBUG
		std::cout << "My interpolation matrix:" << std::endl;
		temporal_interpolation_matrix.print_formatted(std::cout);
#endif

		// NOTE: Alternatively there is most likely also a way
		//       to get the temporal restriction/interpolation matrix in deal.II

		//////////////////////////////
		// space-time sparsity pattern
		DynamicSparsityPattern dsp(displacement_n_dofs + pressure_n_dofs);
			
		//////////
		// space
		// 
		DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs(), space_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(space_dof_handler, space_dsp);

#ifdef DEBUG
		SparsityPattern space_sparsity_pattern;
		space_sparsity_pattern.copy_from(space_dsp);
		std::ofstream out_space_sparsity("space_sparsity_pattern.svg");
		space_sparsity_pattern.print_svg(out_space_sparsity);
#endif

		//////////
		// time
		//

		// displacement
		DynamicSparsityPattern displacement_time_dsp(slab->displacement_time_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(slab->displacement_time_dof_handler, displacement_time_dsp);
		SparsityPattern displacement_time_sparsity_pattern; // NOTE: This temporal sparsity pattern is only to compute the interface sparsity pattern. Therefore, we do not need jump terms here.
		
		// include jump terms in temporal sparsity pattern
		// for Gauss-Legendre quadrature we need to couple all temporal DoFs between two neighboring time intervals
		unsigned int displacement_time_block_size = slab->displacement_time_fe.degree + 1;
		for (unsigned int k = 1; k < slab->displacement_time_triangulation.n_active_cells(); ++k)
		for (unsigned int ii = 0; ii < displacement_time_block_size; ++ii)
			for (unsigned int jj = 0; jj < displacement_time_block_size; ++jj)
				displacement_time_dsp.add(k*displacement_time_block_size+ii, (k-1)*displacement_time_block_size+jj);

		displacement_time_sparsity_pattern.copy_from(displacement_time_dsp);
		
		// add space-time sparsity pattern for (displacement,displacement)-block
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() < displacement_n_space_dofs) && (space_entry.column() < displacement_n_space_dofs))
			for (auto &time_entry : displacement_time_dsp)
				dsp.add(
					space_entry.row()    + displacement_n_space_dofs * time_entry.row(),	// test  function
					space_entry.column() + displacement_n_space_dofs * time_entry.column() // trial function
				);

		// pressure
		DynamicSparsityPattern pressure_time_dsp(slab->pressure_time_dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(slab->pressure_time_dof_handler, pressure_time_dsp);
		SparsityPattern pressure_time_sparsity_pattern; // NOTE: This temporal sparsity pattern is only to compute the interface sparsity pattern. Therefore, we do not need jump terms here.
		
		// include jump terms in temporal sparsity pattern
		// for Gauss-Legendre quadrature we need to couple all temporal DoFs between two neighboring time intervals
		unsigned int pressure_time_block_size = slab->pressure_time_fe.degree + 1;
		for (unsigned int k = 1; k < slab->pressure_time_triangulation.n_active_cells(); ++k)
		for (unsigned int ii = 0; ii < pressure_time_block_size; ++ii)
			for (unsigned int jj = 0; jj < pressure_time_block_size; ++jj)
			pressure_time_dsp.add(k*pressure_time_block_size+ii, (k-1)*pressure_time_block_size+jj);

		pressure_time_sparsity_pattern.copy_from(pressure_time_dsp);

		// add space-time sparsity pattern for (pressure,pressure)-block 
		// Note: offset row and column entries by number of displacement DoFs
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() >= displacement_n_space_dofs) && (space_entry.column() >= displacement_n_space_dofs))
			for (auto &time_entry : pressure_time_dsp)
				dsp.add(
					(space_entry.row()-displacement_n_space_dofs)    + pressure_n_space_dofs * time_entry.row()    + displacement_n_dofs,   // test  function
					(space_entry.column()-displacement_n_space_dofs) + pressure_n_space_dofs * time_entry.column() + displacement_n_dofs    // trial function
				);

		// interface terms:

		// displacement-pressure
		DynamicSparsityPattern displacement_pressure_time_dsp(
			slab->displacement_time_dof_handler.n_dofs(),
			slab->pressure_time_dof_handler.n_dofs()
		);
		if (slab->displacement_time_dof_handler.n_dofs() <= slab->pressure_time_dof_handler.n_dofs())
			displacement_pressure_time_dsp.compute_Tmmult_pattern(
				temporal_interpolation_sparsity_pattern,
				pressure_time_sparsity_pattern
			);
		else
			displacement_pressure_time_dsp.compute_mmult_pattern(
				displacement_time_sparsity_pattern,
				temporal_interpolation_sparsity_pattern
			);
		
#ifdef DEBUG
		SparsityPattern displacement_pressure_time_sparsity_pattern;
		displacement_pressure_time_sparsity_pattern.copy_from(displacement_pressure_time_dsp);
		std::ofstream out_displacement_pressure_time_sparsity("displacement_pressure_time_sparsity_pattern.svg");
		displacement_pressure_time_sparsity_pattern.print_svg(out_displacement_pressure_time_sparsity);
#endif

		// add space-time sparsity pattern for (displacement,pressure)-block
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() < displacement_n_space_dofs) && (space_entry.column() >= displacement_n_space_dofs))
			for (auto &time_entry : displacement_pressure_time_dsp)
				dsp.add(
					space_entry.row()                         + displacement_n_space_dofs * time_entry.row()    + 0,	        // test  function
					(space_entry.column()-displacement_n_space_dofs) + pressure_n_space_dofs * time_entry.column() + displacement_n_dofs // trial function
				);

		// pressure-displacement
		DynamicSparsityPattern pressure_displacement_time_dsp(
			slab->pressure_time_dof_handler.n_dofs(),
			slab->displacement_time_dof_handler.n_dofs()
		);
		if (slab->displacement_time_dof_handler.n_dofs() > slab->pressure_time_dof_handler.n_dofs())
			pressure_displacement_time_dsp.compute_Tmmult_pattern(
				temporal_interpolation_sparsity_pattern,
				displacement_time_sparsity_pattern
			);
		else
			pressure_displacement_time_dsp.compute_mmult_pattern(
				pressure_time_sparsity_pattern,
				temporal_interpolation_sparsity_pattern
			);

#ifdef DEBUG		
		SparsityPattern pressure_displacement_time_sparsity_pattern;
		pressure_displacement_time_sparsity_pattern.copy_from(pressure_displacement_time_dsp);
		std::ofstream out_pressure_displacement_time_sparsity("pressure_displacement_time_sparsity_pattern.svg");
		pressure_displacement_time_sparsity_pattern.print_svg(out_pressure_displacement_time_sparsity);
#endif

		// add space-time sparsity pattern for (pressure,displacement)-block
		for (auto &space_entry : space_dsp)
		if ((space_entry.row() >= displacement_n_space_dofs) && (space_entry.column() < displacement_n_space_dofs))
			for (auto &time_entry : pressure_displacement_time_dsp)
				dsp.add(
					(space_entry.row()-displacement_n_space_dofs) + pressure_n_space_dofs * time_entry.row()    + displacement_n_dofs, // test  function
					space_entry.column()                   + displacement_n_space_dofs * time_entry.column() + 0             // trial function
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
	
	solution.reinit(displacement_n_dofs + pressure_n_dofs);
	system_rhs.reinit(displacement_n_dofs + pressure_n_dofs);
}

template<int dim>
void SpaceTime<dim>::assemble_system(std::shared_ptr<Slab> &slab, bool assemble_matrix) {
	system_matrix = 0;
	system_rhs    = 0;

	Tensor <2,dim> identity;
	for (unsigned int k=0; k<dim; ++k)
		identity[k][k] = 1.;

	////////////////
	// displacement
	//
	{
#ifdef DEBUG
		// check that the entries for the (displacement, displacement) block are being distributed correctly
		DynamicSparsityPattern displacement_dsp(displacement_n_dofs+pressure_n_dofs, displacement_n_dofs+pressure_n_dofs);
#endif

		// space
		QGauss<dim>   space_quad_formula(space_fe.degree + 2);
		QGauss<dim-1> space_face_quad_formula(space_fe.degree + 2);

		FEValues<dim> space_fe_values(space_fe, space_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
		
		FEFaceValues<dim>  space_fe_face_values(space_fe, space_face_quad_formula,
				update_values | update_gradients | update_normal_vectors | update_JxW_values);

		// time
		QGauss<1> time_quad_formula(slab->displacement_time_fe.degree + 2);
		FEValues<1> time_fe_values(slab->displacement_time_fe, time_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int time_dofs_per_cell = slab->displacement_time_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
		std::vector<types::global_dof_index> time_prev_local_dof_indices(time_dofs_per_cell);

		// time FEValues for t_m^+ on current time interval I_m
		// FEValues<1> time_fe_face_values(slab->displacement_time_fe, Quadrature<1>({Point<1>(0.)}), update_values); // using left box rule quadrature
		// time FEValues for t_m^- on previous time interval I_{m-1}
		// FEValues<1> time_prev_fe_face_values(slab->displacement_time_fe, Quadrature<1>({Point<1>(1.)}), update_values); // using right box rule quadrature
	
		// local contributions on space-time cell
		FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		// FullMatrix<double> cell_jump(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

		// locally assemble on each space-time cell
		for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
			space_fe_values.reinit(space_cell);
			space_cell->get_dof_indices(space_local_dof_indices);

			for (const auto &time_cell : slab->displacement_time_dof_handler.active_cell_iterators()) {
				time_fe_values.reinit(time_cell);
				time_cell->get_dof_indices(time_local_dof_indices);
				
				cell_matrix = 0;
				cell_rhs = 0;
				// cell_jump = 0;

				for (const unsigned int qq : time_fe_values.quadrature_point_indices())
				{
					// // time quadrature point
					// const double t_qq = time_fe_values.quadrature_point(qq)[0];
					// right_hand_side.set_time(t_qq);

					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						// // space quadrature point
						// const auto x_q = space_fe_values.quadrature_point(q);

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								// // right hand side
								// cell_rhs(i + ii * space_dofs_per_cell) += (space_fe_values[fluid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
								// 										   right_hand_side.value(x_q, fluid_space_fe.system_to_component_index(i).first) *    // g(t_qq, x_q)
								// 										   space_fe_values.JxW(q) * time_fe_values.JxW(qq)								   // d(t,x)
								// );

								const Tensor<2, dim> gradient_i = space_fe_values[displacement].gradient(i, q);
								double divergence_i = gradient_i[0][0] + gradient_i[1][1];
								const Tensor<2, dim> stress_tensor_i = (
									lame_coefficient_mu * (gradient_i + transpose(gradient_i))
									 + lame_coefficient_lambda * divergence_i * identity
								);

								// system matrix
								if (assemble_matrix)
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_matrix(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												scalar_product(
													stress_tensor_i * time_fe_values.shape_value(ii, qq),							  //  σ(ϕ^u_{i,ii}(t_qq, x_q))
													space_fe_values[displacement].gradient(j, q) * time_fe_values.shape_value(jj, qq) // ∇_x ϕ^u_{j,jj}(t_qq, x_q)
												)
											) *
											space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)
							}
					}
				}

				// // assemble jump terms in system matrix and intial condition in RHS
				// // jump terms: ([v]_m,φ_m^+)_Ω = (v_m^+,φ_m^+)_Ω - (v_m^-,φ_m^+)_Ω = (A) - (B)
				// time_fe_face_values.reinit(time_cell);
				
				// // first we assemble (A): (v_m^+,φ_m^+)_Ω
				// if (assemble_matrix)
				// 	for (const unsigned int q : space_fe_values.quadrature_point_indices())
				// 		for (const unsigned int i : space_fe_values.dof_indices())
				// 			for (const unsigned int ii : time_fe_values.dof_indices())
				// 				for (const unsigned int j : space_fe_values.dof_indices())
				// 					for (const unsigned int jj : time_fe_values.dof_indices())
				// 						cell_matrix(
				// 							j + jj * space_dofs_per_cell,
				// 							i + ii * space_dofs_per_cell
				// 						) += (
				// 							space_fe_values[fluid_velocity].value(i, q) * time_fe_face_values.shape_value(ii, 0) * //  ϕ^v_{i,ii}(t_m^+, x_q)
				// 							space_fe_values[fluid_velocity].value(j, q) * time_fe_face_values.shape_value(jj, 0)   //  ϕ^v_{j,jj}(t_m^+, x_q)
				// 						) * space_fe_values.JxW(q); 						//  d(x)

				// // initial condition and jump terms
				// if (time_cell->active_cell_index() == 0)
				// {
				// 	//////////////////////////
				// 	// initial condition

				// 	// (v_0^-,φ_0^-)_Ω
				// 	for (const unsigned int q : space_fe_values.quadrature_point_indices())
				// 	{
				// 		double initial_solution_v_x_q = 0.;
				// 		for (const unsigned int j : space_fe_values.dof_indices())
				// 		{
				// 			initial_solution_v_x_q += initial_solution_fluid[space_local_dof_indices[j]] * space_fe_values[fluid_velocity].value(j, q);
				// 		}

				// 		for (const unsigned int i : space_fe_values.dof_indices())
				// 			for (const unsigned int ii : time_fe_values.dof_indices())
				// 			{
				// 				cell_rhs(i + ii * space_dofs_per_cell) += (initial_solution_v_x_q *										   // v0(x_q)
				// 														space_fe_values[fluid_velocity].value(i, q) * time_fe_face_values.shape_value(ii, 0) * // ϕ^v_{i,ii}(0^+, x_q)
				// 														space_fe_values.JxW(q)															// d(x)
				// 				);
				// 			}
				// 	}
				// }
				// else
				// {
				// 	//////////////
				// 	// jump term

				// 	// now we assemble (B): - (u_m^-,φ_m^+)_Ω
				// 	// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre quadrature in time
				// 	if (assemble_matrix)
				// 		for (const unsigned int q : space_fe_values.quadrature_point_indices())
				// 			for (const unsigned int i : space_fe_values.dof_indices())
				// 				for (const unsigned int ii : time_fe_values.dof_indices())
				// 					for (const unsigned int j : space_fe_values.dof_indices())
				// 						for (const unsigned int jj : time_fe_values.dof_indices())
				// 							cell_jump(
				// 								j + jj * space_dofs_per_cell,
				// 								i + ii * space_dofs_per_cell
				// 							) += (
				// 								-1. * space_fe_values[fluid_velocity].value(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -ϕ^v_{i,ii}(t_m^-, x_q)
				// 								space_fe_values[fluid_velocity].value(j, q) * time_fe_face_values.shape_value(jj, 0)			  //  ϕ^v_{j,jj}(t_m^+, x_q)
				// 							) * space_fe_values.JxW(q); //  d(x)
				// }

				// distribute local to global
				for (const unsigned int i : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
					{
						// // right hand side
						// system_rhs(space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs) += cell_rhs(i + ii * space_dofs_per_cell);

						// system matrix
						if (assemble_matrix)
							for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
								{
									system_matrix.add(
										space_local_dof_indices[i] + time_local_dof_indices[ii] * displacement_n_space_dofs,
										space_local_dof_indices[j] + time_local_dof_indices[jj] * displacement_n_space_dofs,
										cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));
#ifdef DEBUG						
									if (cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell) != 0.)
										displacement_dsp.add(
											space_local_dof_indices[i] + time_local_dof_indices[ii] * displacement_n_space_dofs,
											space_local_dof_indices[j] + time_local_dof_indices[jj] * displacement_n_space_dofs
										);
#endif
								}
					}

// 				// distribute cell jump
// 				if (assemble_matrix)
// 					if (time_cell->active_cell_index() > 0)
// 						for (const unsigned int i : space_fe_values.dof_indices())
// 							for (const unsigned int ii : time_fe_values.dof_indices())
// 								for (const unsigned int j : space_fe_values.dof_indices())
// 									for (const unsigned int jj : time_fe_values.dof_indices())
// 									{
// 										system_matrix.add(
// 											space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
// 											space_local_dof_indices[j] + time_prev_local_dof_indices[jj] * fluid_n_space_dofs,
// 											cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

// #ifdef DEBUG
// 										fluid_dsp.add(
// 											space_local_dof_indices[i] + time_local_dof_indices[ii] * fluid_n_space_dofs,
// 											space_local_dof_indices[j] + time_prev_local_dof_indices[jj] * fluid_n_space_dofs
// 										);
// #endif
// 									}

// 				// prepare next time cell
// 				if (time_cell->active_cell_index() < slab->fluid_time_triangulation.n_active_cells() - 1)
// 				{
// 					time_prev_fe_face_values.reinit(time_cell);
// 					time_cell->get_dof_indices(time_prev_local_dof_indices);
// 				}
			}

			// boundary terms for (displacement,displacement)
			// traction term on top boundary
			for (const unsigned int space_face : space_cell->face_indices())
				if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 3)) // face is at top boundary
				{
					space_fe_face_values.reinit(space_cell, space_face);
					for (const auto &time_cell : slab->displacement_time_dof_handler.active_cell_iterators()) {
						time_fe_values.reinit(time_cell);
						time_cell->get_dof_indices(time_local_dof_indices);

						cell_rhs = 0;

						Tensor<1,dim> neumann_value;
						neumann_value[0] = traction_x_biot;
						neumann_value[1] = traction_y_biot;

						for (const unsigned int qq : time_fe_values.quadrature_point_indices())
						for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
							for (const unsigned int i : space_fe_face_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								cell_rhs(
									i + ii * space_dofs_per_cell
								) += (
									neumann_value * space_fe_face_values[displacement].value(i, q) * time_fe_values.shape_value(ii, qq) // t * ϕ^u_{i,ii}(t_qq, x_q)
								) * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							// d(t,x)

						// distribute local to global
						for (const unsigned int i : space_fe_face_values.dof_indices())
						for (const unsigned int ii : time_fe_values.dof_indices())
						{
							system_rhs(
								space_local_dof_indices[i] + time_local_dof_indices[ii] * displacement_n_space_dofs
							) += cell_rhs(i + ii * space_dofs_per_cell);
						}
					}
				}
		}

#ifdef DEBUG
		if (assemble_matrix)
		{
			SparsityPattern displacement_sparsity_pattern;
			displacement_sparsity_pattern.copy_from(displacement_dsp);
			std::ofstream out_displacement_sparsity("displacement_sparsity_pattern.svg");
			displacement_sparsity_pattern.print_svg(out_displacement_sparsity);
		}
#endif
	}
	
	////////////
	// pressure
	//
	{
#ifdef DEBUG
		// check that the entries for the (pressure, pressure) block are being distributed correctly
		DynamicSparsityPattern pressure_dsp(displacement_n_dofs+pressure_n_dofs, displacement_n_dofs+pressure_n_dofs);
#endif

		// space
		QGauss<dim> space_quad_formula(space_fe.degree + 2);
		// QGauss<dim-1> space_face_quad_formula(space_fe.degree + 2);

		FEValues<dim> space_fe_values(space_fe, space_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);
		
		// FEFaceValues<dim>  space_fe_face_values(space_fe, space_face_quad_formula,
		// 		update_values | update_gradients | update_normal_vectors | update_JxW_values);

		// time
		QGauss<1> time_quad_formula(slab->pressure_time_fe.degree + 2);
		FEValues<1> time_fe_values(slab->pressure_time_fe, time_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int time_dofs_per_cell = slab->pressure_time_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
		std::vector<types::global_dof_index> time_prev_local_dof_indices(time_dofs_per_cell);

		// time FEValues for t_m^+ on current time interval I_m
		FEValues<1> time_fe_face_values(slab->pressure_time_fe, Quadrature<1>({Point<1>(0.)}), update_values); // using left box rule quadrature
		// time FEValues for t_m^- on previous time interval I_{m-1}
		FEValues<1> time_prev_fe_face_values(slab->pressure_time_fe, Quadrature<1>({Point<1>(1.)}), update_values); // using right box rule quadrature

		// local contributions on space-time cell
		FullMatrix<double> cell_matrix(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		FullMatrix<double> cell_jump(space_dofs_per_cell * time_dofs_per_cell, space_dofs_per_cell * time_dofs_per_cell);
		Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices(space_dofs_per_cell * time_dofs_per_cell);

		// locally assemble on each space-time cell
		for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
			space_fe_values.reinit(space_cell);
			space_cell->get_dof_indices(space_local_dof_indices);

			for (const auto &time_cell : slab->pressure_time_dof_handler.active_cell_iterators()) {
				time_fe_values.reinit(time_cell);
				time_cell->get_dof_indices(time_local_dof_indices);
				
				cell_matrix = 0;
				cell_rhs = 0;
				cell_jump = 0;

				for (const unsigned int qq : time_fe_values.quadrature_point_indices())
				{
					// // time quadrature point
					// const double t_qq = time_fe_values.quadrature_point(qq)[0];
					// right_hand_side.set_time(t_qq);

					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						// // space quadrature point
						// const auto x_q = space_fe_values.quadrature_point(q);

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								// // right hand side
								// cell_rhs(i + ii * space_dofs_per_cell) += (
								// 	space_fe_values[solid_velocity].value(i, q) * time_fe_values.shape_value(ii, qq) * // ϕ_{i,ii}(t_qq, x_q)
								// 	right_hand_side.value(x_q, solid_space_fe.system_to_component_index(i).first) *	   // g(t_qq, x_q)
								// 	space_fe_values.JxW(q) * time_fe_values.JxW(qq)								// d(t,x)
								// );

								// system matrix
								if (assemble_matrix)
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_matrix(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												c_biot * 																		// c
												space_fe_values[pressure].value(i, q) * time_fe_values.shape_grad(ii, qq)[0] *	// ∂_t ϕ^p_{i,ii}(t_qq, x_q)
												space_fe_values[pressure].value(j, q) * time_fe_values.shape_value(jj, qq)		//     ϕ^p_{j,jj}(t_qq, x_q)
																														// +
												+ (K_biot / viscosity_biot) * 													// (K /  ν)
												space_fe_values[pressure].gradient(i, q) * time_fe_values.shape_value(ii, qq) * // ∇_x ϕ^p_{i,ii}(t_qq, x_q)
												space_fe_values[pressure].gradient(j, q) * time_fe_values.shape_value(jj, qq)	// ∇_x ϕ^p_{j,jj}(t_qq, x_q)
											) * space_fe_values.JxW(q) * time_fe_values.JxW(qq); // d(t,x)
							}
					}
				}

				// assemble jump terms in system matrix and intial condition in RHS
				// jump terms: 
				//    a) for pressure: c([p]_m,ϕ^{p,+}_m)_Ω = c(p_m^+,ϕ^{p,+}_m)_Ω - c(p_m^-,ϕ^{p,+}_m)_Ω = (A1) - (B1)
				time_fe_face_values.reinit(time_cell);

				// first we assemble (A1): c(p_m^+,ϕ^{p,+}_m)_Ω
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
											c_biot * 																		 // c
											space_fe_values[pressure].value(i, q) * time_fe_face_values.shape_value(ii, 0) * //  ϕ^p_{i,ii}(t_m^+, x_q)
											space_fe_values[pressure].value(j, q) * time_fe_face_values.shape_value(jj, 0)   //  ϕ^p_{j,jj}(t_m^+, x_q)
										) * space_fe_values.JxW(q); //  d(x)

				// initial condition and jump terms
				if (time_cell->active_cell_index() == 0)
				{
					//////////////////////////
					// initial condition

					// std::vector<Vector<double>> initial_solution_values(
            		// 	space_quad_formula.size(), Vector<double>(2 + 1));
					// space_fe_values.get_function_values(initial_solution_pressure,
                    //                         initial_solution_values);

					// c(p_0^-,ϕ^{p,+}_0)_Ω
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						double initial_solution_p_x_q = 0.;
						for (const unsigned int j : space_fe_values.dof_indices())
						{
							initial_solution_p_x_q += initial_solution_pressure[space_local_dof_indices[j]] * space_fe_values[pressure].value(j, q);
						}

						// std::cout << "  p(0) = " << initial_solution_p_x_q << std::endl;
						// std::cout << "  p_new(0) = " << initial_solution_values[q](2) << std::endl;

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								cell_rhs(i + ii * space_dofs_per_cell) += (
									c_biot * 																	   // c
									initial_solution_p_x_q *                                             		   //   p_0(x_q)
									space_fe_values[pressure].value(i, q) * time_fe_face_values.shape_value(ii, 0) // ϕ^p_{i,ii}(0^+, x_q)
								) * space_fe_values.JxW(q);   // d(x)
							}
					}
				}
				else
				{
					//////////////
					// jump term

					// now we assemble (B1): - c(p_m^-,ϕ^{p,+}_m)_Ω
					// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre or any other arbitrary quadrature in time
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
												c_biot * 																		            // c
												-1. * space_fe_values[pressure].value(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -ϕ^p_{i,ii}(t_m^-, x_q)
												space_fe_values[pressure].value(j, q) * time_fe_face_values.shape_value(jj, 0)              //  ϕ^p_{j,jj}(t_m^+, x_q)
											) * space_fe_values.JxW(q); 		      //  d(x)
				}

				// distribute local to global (NOTE: need to offset rows and columns by number of displacement space-time DoFs)
				for (const unsigned int i : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
					{
						// right hand side
						system_rhs((space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs) += cell_rhs(i + ii * space_dofs_per_cell);

						// system matrix
						if (assemble_matrix)
							for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
								{
									system_matrix.add(
										(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs,
										(space_local_dof_indices[j]-displacement_n_space_dofs) + time_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs,
										cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

#ifdef DEBUG
									if (cell_matrix(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell) != 0.)
										pressure_dsp.add(
											(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs,
											(space_local_dof_indices[j]-displacement_n_space_dofs) + time_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs
										);
#endif
								}
					}

				// distribute cell jump (NOTE: need to offset rows and columns by number of displacement space-time DoFs)
				if (assemble_matrix)
					if (time_cell->active_cell_index() > 0)
						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
									{
										system_matrix.add(
											(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs      + displacement_n_dofs,
											(space_local_dof_indices[j]-displacement_n_space_dofs) + time_prev_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs,
											cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell));

#ifdef DEBUG
										if (cell_jump(i + ii * space_dofs_per_cell, j + jj * space_dofs_per_cell) != 0.)
											pressure_dsp.add(
												(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs      + displacement_n_dofs,
												(space_local_dof_indices[j]-displacement_n_space_dofs) + time_prev_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs
											);
#endif
								}

				// prepare next time cell
				if (time_cell->active_cell_index() < slab->pressure_time_triangulation.n_active_cells() - 1)
				{
					time_prev_fe_face_values.reinit(time_cell);
					time_cell->get_dof_indices(time_prev_local_dof_indices);
				}
			}

// 			// interface terms for (solid,solid)
// 			if (assemble_matrix)
// 				for (const unsigned int space_face : space_cell->face_indices())
// 					if (space_cell->at_boundary(space_face) == false) // face is not at boundary
// 					if (space_cell->neighbor(space_face)->material_id() == fluid_domain_id) // face is at interface (= fluid & solid cell meet)
// 					{
// 						space_fe_face_values.reinit(space_cell, space_face);
// 						for (const auto &time_cell : slab->solid_time_dof_handler.active_cell_iterators()) {
// 							time_fe_values.reinit(time_cell);
// 							time_cell->get_dof_indices(time_local_dof_indices);

// 							cell_matrix = 0;

// 							for (const unsigned int qq : time_fe_values.quadrature_point_indices())
// 							for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
// 								for (const unsigned int i : space_fe_face_values.dof_indices())
// 								for (const unsigned int ii : time_fe_values.dof_indices())
// 									for (const unsigned int j : space_fe_face_values.dof_indices())
// 									for (const unsigned int jj : time_fe_values.dof_indices())
// 										cell_matrix(
// 											j + jj * space_dofs_per_cell,
// 											i + ii * space_dofs_per_cell
// 										) += (
// 											-delta * // -δ
// 											space_fe_face_values[solid_velocity].gradient(i, q) * space_fe_face_values.normal_vector(q) * time_fe_values.shape_value(ii, qq) * // ∇_x ϕ^v_{i,ii}(t_qq, x_q) · n_s(x_q)
// 											space_fe_face_values[solid_velocity].value(j, q) * time_fe_values.shape_value(jj, qq)                                              //     ϕ^v_{j,jj}(t_qq, x_q)
// 										) * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							// d(t,x)

// 							// distribute local to global (NOTE: need to offset rows and columns by number of fluid space-time DoFs)
// 							for (const unsigned int i : space_fe_face_values.dof_indices())
// 							for (const unsigned int ii : time_fe_values.dof_indices())
// 								for (const unsigned int j : space_fe_values.dof_indices())
// 								for (const unsigned int jj : time_fe_values.dof_indices())
// 								{
// 									system_matrix.add(
// 										(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
// 										(space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs,
// 										cell_matrix(
// 											i + ii * space_dofs_per_cell,
// 											j + jj * space_dofs_per_cell
// 										)
// 									);

// #ifdef DEBUG
// 									solid_dsp.add(
// 										(space_local_dof_indices[i]-fluid_n_space_dofs) + time_local_dof_indices[ii] * solid_n_space_dofs + fluid_n_dofs,
// 										(space_local_dof_indices[j]-fluid_n_space_dofs) + time_local_dof_indices[jj] * solid_n_space_dofs + fluid_n_dofs
// 									);
// #endif
// 								}
// 						}
// 					}
		}

#ifdef DEBUG
		if (assemble_matrix)
		{
			SparsityPattern pressure_sparsity_pattern;
			pressure_sparsity_pattern.copy_from(pressure_dsp);
			std::ofstream out_pressure_sparsity("pressure_sparsity_pattern.svg");
			pressure_sparsity_pattern.print_svg(out_pressure_sparsity);
		}
#endif
	}

	////////////
	// coupling
	//
	{
#ifdef DEBUG
		// create the sparsity pattern that is required for the coupling terms
		DynamicSparsityPattern coupling_dsp(displacement_n_dofs+pressure_n_dofs, displacement_n_dofs+pressure_n_dofs);
#endif

		// space:
		QGauss<dim>   space_quad_formula(space_fe.degree + 2);
		QGauss<dim-1> space_face_quad_formula(space_fe.degree + 2);

		FEValues<dim> space_fe_values(space_fe, space_quad_formula,
				update_values | update_gradients | update_quadrature_points | update_JxW_values);
		FEFaceValues<dim> space_fe_face_values(space_fe, space_face_quad_formula,
			update_values | update_gradients | update_normal_vectors | update_JxW_values);

		const unsigned int space_dofs_per_cell = space_fe.n_dofs_per_cell();
		std::vector<types::global_dof_index> space_local_dof_indices(space_dofs_per_cell);

		// time:
		//   using temporal FE with finer temporal mesh
		bool displacement_is_finer = (slab->displacement_time_dof_handler.n_dofs() > slab->pressure_time_dof_handler.n_dofs());
		auto time_fe = (
			displacement_is_finer ?
				&(slab->displacement_time_fe) :
				&(slab->pressure_time_fe)
		);
		auto time_dof_handler = (
			displacement_is_finer ?
				&(slab->displacement_time_dof_handler) :
				&(slab->pressure_time_dof_handler)
		);
		auto time_triangulation = (
			displacement_is_finer ?
				&(slab->displacement_time_triangulation) :
				&(slab->pressure_time_triangulation)
		);
		QGauss<1> time_quad_formula(time_fe->degree + 2);
		FEValues<1> time_fe_values(*time_fe, time_quad_formula,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
		const unsigned int time_dofs_per_cell = time_fe->n_dofs_per_cell();
		std::vector<types::global_dof_index> time_local_dof_indices(time_dofs_per_cell);
		std::vector<types::global_dof_index> time_prev_local_dof_indices(time_dofs_per_cell);

		// time FEValues for t_m^+ on current time interval I_m
		FEValues<1> time_fe_face_values(*time_fe, Quadrature<1>({Point<1>(0.)}), update_values); // using left box rule quadrature
		// time FEValues for t_m^- on previous time interval I_{m-1}
		FEValues<1> time_prev_fe_face_values(*time_fe, Quadrature<1>({Point<1>(1.)}), update_values); // using right box rule quadrature


		// local contributions on space-time cell
		FullMatrix<double> cell_matrix_pressure_displacement(
			space_dofs_per_cell * time_dofs_per_cell,
			space_dofs_per_cell * time_dofs_per_cell
		);
		FullMatrix<double> cell_jump_pressure_displacement(
			space_dofs_per_cell * time_dofs_per_cell, 
			space_dofs_per_cell * time_dofs_per_cell
		);
		Vector<double> cell_rhs(space_dofs_per_cell * time_dofs_per_cell);

		FullMatrix<double> cell_matrix_displacement_pressure(
			space_dofs_per_cell * time_dofs_per_cell,
			space_dofs_per_cell * time_dofs_per_cell
		);
		// std::vector<types::global_dof_index> displacement_local_dof_indices(displacement_space_dofs_per_cell * time_dofs_per_cell);
		// std::vector<types::global_dof_index> pressure_local_dof_indices(pressure_space_dofs_per_cell * time_dofs_per_cell);

		// locally assemble on each space-time cell
		for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
			space_fe_values.reinit(space_cell);
			space_cell->get_dof_indices(space_local_dof_indices);

			for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
				time_fe_values.reinit(time_cell);
				time_cell->get_dof_indices(time_local_dof_indices);
				
				cell_matrix_pressure_displacement = 0;
				cell_jump_pressure_displacement = 0;
				cell_rhs = 0;
				cell_matrix_displacement_pressure = 0;

				if (assemble_matrix)
					for (const unsigned int qq : time_fe_values.quadrature_point_indices())
					{
// #ifdef DEBUG
// 						const double t_qq = time_fe_values.quadrature_point(qq)[0];
// 						std::cout << " t_qq = " << t_qq << std::endl;
// #endif
						for (const unsigned int q : space_fe_values.quadrature_point_indices())
						{
// #ifdef DEBUG
// 							const auto x_q = space_fe_values.quadrature_point(q);
// 							std::cout << " x_q = (" << x_q[0] << "," << x_q[1] << ")" << std::endl;
// #endif
							for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
								for (const unsigned int jj : time_fe_values.dof_indices())
								{
									// -α((pI, ∇_x φ^u))
									cell_matrix_displacement_pressure(
										j + jj * space_dofs_per_cell,
										i + ii * space_dofs_per_cell
									) -= (
										alpha_biot *                                                            								// α
										scalar_product(
											space_fe_values[pressure].value(i, q) * identity * time_fe_values.shape_value(ii, qq),      //     ϕ^p_{i,ii}(t_qq, x_q) * I
											space_fe_values[displacement].gradient(j, q) * time_fe_values.shape_value(jj, qq)  			// ∇_x ϕ^u_{j,jj}(t_qq, x_q)
										)
									) * space_fe_values.JxW(q) * time_fe_values.JxW(qq);

									// α((∂_t(∇_x·u), φ^p))
									cell_matrix_pressure_displacement(
										j + jj * space_dofs_per_cell,
										i + ii * space_dofs_per_cell
									) += (
										alpha_biot *                                                             			         // α
										space_fe_values[displacement].divergence(i, q) * time_fe_values.shape_grad(ii, qq)[0] * // ∂_t(∇_x·ϕ^u_{i,ii})(t_qq, x_q)
										space_fe_values[pressure].value(j, q) * time_fe_values.shape_value(jj, qq)              //         ϕ^p_{j,jj}(t_qq, x_q)
									) * space_fe_values.JxW(q) * time_fe_values.JxW(qq); 
								}
						}
					}

				// assemble jump terms in system matrix and intial condition in RHS
				// jump terms: 
				//    a) for divergence of displacement: α([∇_x·u]_m,ϕ^{p,+}_m)_Ω = α(∇_x·u_m^+,ϕ^{p,+}_m)_Ω - α(∇_x·u_m^-,ϕ^{p,+}_m)_Ω = (A1) - (B1)
				time_fe_face_values.reinit(time_cell);

				// first we assemble (A1): α(∇_x·u_m^+,ϕ^{p,+}_m)_Ω
				if (assemble_matrix)
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
								for (const unsigned int j : space_fe_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
										cell_matrix_pressure_displacement(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell
										) += (
											alpha_biot * 																	 		  // α
											space_fe_values[displacement].divergence(i, q) * time_fe_face_values.shape_value(ii, 0) * // ∇_x·ϕ^u_{i,ii}(t_m^+, x_q)
											space_fe_values[pressure].value(j, q) * time_fe_face_values.shape_value(jj, 0)    		  //     ϕ^p_{j,jj}(t_m^+, x_q)
										) * space_fe_values.JxW(q); //  d(x)
				
				// initial condition and jump terms
				if (time_cell->active_cell_index() == 0)
				{
					//////////////////////////
					// initial condition

					// std::vector<std::vector<Tensor<1, dim>>> initial_solution_grads(
					// 	space_quad_formula.size(), std::vector<Tensor<1, dim>>(2 + 1));
					// space_fe_values.get_function_gradients(initial_solution_displacement,
					// 									initial_solution_grads);

					// α(∇_x·u_0^-,ϕ^{p,+}_0)_Ω
					for (const unsigned int q : space_fe_values.quadrature_point_indices())
					{
						double initial_solution_div_u_x_q = 0.;
						for (const unsigned int j : space_fe_values.dof_indices())
						{
							initial_solution_div_u_x_q += initial_solution_displacement[space_local_dof_indices[j]] * space_fe_values[displacement].divergence(j, q);
						}

						// std::cout << "  div(u(0)) = " << initial_solution_div_u_x_q << std::endl;
						// std::cout << "  div_new(u(0)) = " << initial_solution_grads[q][0][0] + initial_solution_grads[q][1][1] << std::endl;

						for (const unsigned int i : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							{
								cell_rhs(i + ii * space_dofs_per_cell) += (
									alpha_biot * 																   // α
									initial_solution_div_u_x_q *                                             	   // ∇_x·u_0(x_q)
									space_fe_values[pressure].value(i, q) * time_fe_face_values.shape_value(ii, 0) //   ϕ^p_{i,ii}(0^+, x_q)
								) * space_fe_values.JxW(q);   // d(x)
							}
					}
				}
				else
				{
					//////////////
					// jump term

					// now we assemble (B1): - α(∇_x·u_m^-,ϕ^{p,+}_m)_Ω
					// NOTE: cell_jump is a space-time cell matrix because we are using Gauss-Legendre or any other arbitrary quadrature in time
					if (assemble_matrix)
						for (const unsigned int q : space_fe_values.quadrature_point_indices())
							for (const unsigned int i : space_fe_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_values.dof_indices())
										for (const unsigned int jj : time_fe_values.dof_indices())
											cell_jump_pressure_displacement(
												j + jj * space_dofs_per_cell,
												i + ii * space_dofs_per_cell
											) += (
												alpha_biot * 																		                 // α
												-1. * space_fe_values[displacement].divergence(i, q) * time_prev_fe_face_values.shape_value(ii, 0) * // -∇_x·ϕ^u_{i,ii}(t_m^-, x_q)
												space_fe_values[pressure].value(j, q) * time_fe_face_values.shape_value(jj, 0)                       //      ϕ^p_{j,jj}(t_m^+, x_q)
											) * space_fe_values.JxW(q); 		      //  d(x)
				}

				// distribute local to global for (pressure, displacement) to system matrix
				if (assemble_matrix)
				for (const unsigned int i : space_fe_values.dof_indices())
				for (const unsigned int j : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
					for (const unsigned int jj : time_fe_values.dof_indices())
						for (unsigned int kk = 0; kk < temporal_interpolation_matrix.n(); ++kk)
						{
#ifdef DEBUG
							if (!displacement_is_finer)
							{
								if (
									cell_matrix_pressure_displacement(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk) != 0.
								)
									coupling_dsp.add(
										(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs,
										space_local_dof_indices[j]                      + kk * displacement_n_space_dofs                         + 0
									);
							}
							else
							{
								if (
									cell_matrix_pressure_displacement(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk) != 0.
								)
									coupling_dsp.add(
										(space_local_dof_indices[i]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs,
										space_local_dof_indices[j]                      + time_local_dof_indices[jj] * displacement_n_space_dofs + 0
									);
							}
#endif

							if (!displacement_is_finer)
								system_matrix.add(
									(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs,
									space_local_dof_indices[j]                      + kk * displacement_n_space_dofs                         + 0,
									cell_matrix_pressure_displacement(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk)
								);
							else
								system_matrix.add(
									(space_local_dof_indices[i]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs,
									space_local_dof_indices[j]                      + time_local_dof_indices[jj] * displacement_n_space_dofs + 0,
									cell_matrix_pressure_displacement(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk)
								);

#ifdef DEBUG
							if (time_cell->active_cell_index() > 0)
							{
								if (!displacement_is_finer)
								{
									if (
										cell_jump_pressure_displacement(
											i + ii * space_dofs_per_cell,
											j + jj * space_dofs_per_cell
										) * temporal_interpolation_matrix.el(time_prev_local_dof_indices[jj], kk) != 0.
									)
										coupling_dsp.add(
											(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs,
											space_local_dof_indices[j]                      + kk * displacement_n_space_dofs                         + 0
										);
								}
								else
								{	
									if (
										cell_jump_pressure_displacement(
											i + ii * space_dofs_per_cell,
											j + jj * space_dofs_per_cell
										) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk) != 0.
									)
										coupling_dsp.add(
											(space_local_dof_indices[i]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs,
											space_local_dof_indices[j]                      + time_prev_local_dof_indices[jj] * displacement_n_space_dofs + 0
										);
								}
							}
#endif

							if (time_cell->active_cell_index() > 0)
							{
								if (!displacement_is_finer)
									system_matrix.add(
										(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs,
										space_local_dof_indices[j]                      + kk * displacement_n_space_dofs                         + 0,
										cell_jump_pressure_displacement(
											i + ii * space_dofs_per_cell,
											j + jj * space_dofs_per_cell
										) * temporal_interpolation_matrix.el(time_prev_local_dof_indices[jj], kk)
									);
								else
									system_matrix.add(
										(space_local_dof_indices[i]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs,
										space_local_dof_indices[j]                      + time_prev_local_dof_indices[jj] * displacement_n_space_dofs + 0,
										cell_jump_pressure_displacement(
											i + ii * space_dofs_per_cell,
											j + jj * space_dofs_per_cell
										) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk)
									);
							}
						}

				// distribute local to global for (pressure, displacement) to system rhs
				for (const unsigned int i : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
					{
						if (displacement_is_finer)
						{
							for (unsigned int kk = 0; kk < temporal_interpolation_matrix.n(); ++kk)
							{
								system_rhs(
									(space_local_dof_indices[i]-displacement_n_space_dofs) + kk * pressure_n_space_dofs + displacement_n_dofs
								) += cell_rhs(i + ii * space_dofs_per_cell) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk);
							}
						}
						else
						{
							system_rhs(
								(space_local_dof_indices[i]-displacement_n_space_dofs) + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs
							) += cell_rhs(i + ii * space_dofs_per_cell);
						}
					}
				
				// distribute local to global for (displacement, pressure) to system matrix
				if (assemble_matrix)
				for (const unsigned int i : space_fe_values.dof_indices())
				for (const unsigned int j : space_fe_values.dof_indices())
					for (const unsigned int ii : time_fe_values.dof_indices())
					for (const unsigned int jj : time_fe_values.dof_indices())
						for (unsigned int kk = 0; kk < temporal_interpolation_matrix.n(); ++kk)
						{
#ifdef DEBUG
							if (displacement_is_finer)
							{
								if (
									cell_matrix_displacement_pressure(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk) != 0.
								)
									coupling_dsp.add(
										space_local_dof_indices[i]                     + time_local_dof_indices[ii] * displacement_n_space_dofs + 0,
										(space_local_dof_indices[j]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs
									);
							}
							else
							{
								if (
									cell_matrix_displacement_pressure(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk) != 0.
								)
									coupling_dsp.add(
										space_local_dof_indices[i]                     + kk * displacement_n_space_dofs                         + 0,
										(space_local_dof_indices[j]-displacement_n_space_dofs) + time_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs
									); 
							}
#endif
							if (displacement_is_finer)
								system_matrix.add(
									space_local_dof_indices[i]                     + time_local_dof_indices[ii] * displacement_n_space_dofs + 0,
									(space_local_dof_indices[j]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs,
									cell_matrix_displacement_pressure(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk)
								);
							else
								system_matrix.add(
									space_local_dof_indices[i]                     + kk * displacement_n_space_dofs                         + 0,
									(space_local_dof_indices[j]-displacement_n_space_dofs) + time_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs,
									cell_matrix_displacement_pressure(
										i + ii * space_dofs_per_cell,
										j + jj * space_dofs_per_cell
									) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk)
								); 
						}


				// prepare next time cell
				if (time_cell->active_cell_index() < time_triangulation->n_active_cells() - 1)
				{
					time_prev_fe_face_values.reinit(time_cell);
					time_cell->get_dof_indices(time_prev_local_dof_indices);
				}
			}

			for (const unsigned int space_face : space_cell->face_indices())
				if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 3)) // face is at top boundary
				{
					space_fe_face_values.reinit(space_cell, space_face);
					for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
						time_fe_values.reinit(time_cell);
						time_cell->get_dof_indices(time_local_dof_indices);

						cell_matrix_displacement_pressure = 0;

						// α << pn, φ^u >>
						if (assemble_matrix)
							for (const unsigned int qq : time_fe_values.quadrature_point_indices())
							for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
								for (const unsigned int i : space_fe_face_values.dof_indices())
								for (const unsigned int ii : time_fe_values.dof_indices())
									for (const unsigned int j : space_fe_face_values.dof_indices())
									for (const unsigned int jj : time_fe_values.dof_indices())
										cell_matrix_displacement_pressure(
											j + jj * space_dofs_per_cell,
											i + ii * space_dofs_per_cell
										) += (
											alpha_biot * 																										   // α
											space_fe_values[pressure].value(i, q) * space_fe_face_values.normal_vector(q) * time_fe_values.shape_value(ii, qq) *   // ϕ^p_{i,ii}(t_qq, x_q) · n(x_q)
											space_fe_values[displacement].value(j, q) * time_fe_values.shape_value(jj, qq)  									   // ϕ^u_{j,jj}(t_qq, x_q)
										) * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq); 							// d(t,x)


						// distribute local to global for (displacement, pressure) to system matrix
						if (assemble_matrix)
						for (const unsigned int i : space_fe_values.dof_indices())
						for (const unsigned int j : space_fe_values.dof_indices())
							for (const unsigned int ii : time_fe_values.dof_indices())
							for (const unsigned int jj : time_fe_values.dof_indices())
								for (unsigned int kk = 0; kk < temporal_interpolation_matrix.n(); ++kk)
								{
#ifdef DEBUG
									if (displacement_is_finer)
									{
										if (
											cell_matrix_displacement_pressure(
												i + ii * space_dofs_per_cell,
												j + jj * space_dofs_per_cell
											) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk) != 0.
										)
											coupling_dsp.add(
												space_local_dof_indices[i]                     + time_local_dof_indices[ii] * displacement_n_space_dofs + 0,
												(space_local_dof_indices[j]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs
											);
									}
									else
									{
										if (
											cell_matrix_displacement_pressure(
												i + ii * space_dofs_per_cell,
												j + jj * space_dofs_per_cell
											) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk) != 0.
										)
											coupling_dsp.add(
												space_local_dof_indices[i]                     + kk * displacement_n_space_dofs                         + 0,
												(space_local_dof_indices[j]-displacement_n_space_dofs) + time_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs
											); 
									}
#endif
									if (displacement_is_finer)
										system_matrix.add(
											space_local_dof_indices[i]                     + time_local_dof_indices[ii] * displacement_n_space_dofs + 0,
											(space_local_dof_indices[j]-displacement_n_space_dofs) + kk * pressure_n_space_dofs                         + displacement_n_dofs,
											cell_matrix_displacement_pressure(
												i + ii * space_dofs_per_cell,
												j + jj * space_dofs_per_cell
											) * temporal_interpolation_matrix.el(time_local_dof_indices[jj], kk)
										);
									else
										system_matrix.add(
											space_local_dof_indices[i]                     + kk * displacement_n_space_dofs                         + 0,
											(space_local_dof_indices[j]-displacement_n_space_dofs) + time_local_dof_indices[jj] * pressure_n_space_dofs + displacement_n_dofs,
											cell_matrix_displacement_pressure(
												i + ii * space_dofs_per_cell,
												j + jj * space_dofs_per_cell
											) * temporal_interpolation_matrix.el(time_local_dof_indices[ii], kk)
										); 
								}
					}
				}
		}

#ifdef DEBUG
		SparsityPattern coupling_sparsity_pattern;
		coupling_sparsity_pattern.copy_from(coupling_dsp);
		std::ofstream out_coupling_sparsity("coupling_sparsity_pattern.svg");
		coupling_sparsity_pattern.print_svg(out_coupling_sparsity);
#endif
	}

	apply_boundary_conditions(slab);

#ifdef DEBUG
  if (assemble_matrix) {
    // print out system matrix
    std::ofstream out_matrix("matrix.txt");
    print_as_numpy_arrays_high_resolution(system_matrix, out_matrix,
                                          /*precision*/ 16);
  }
#endif
}	

template<int dim>
void SpaceTime<dim>::apply_boundary_conditions(std::shared_ptr<Slab> &slab) {
	// 4 different walls with boundary ids:
	// 0: left
	// 1: right
	// 2: bottom
	// 3: top

	// Dirichlet boundary conditions are given as:
	// 0: u_x = 0 (Dirichlet displacement;   left)
	// 1:   p = 0 (Dirichlet pressure;      right)
	// 2: u_y = 0 (Dirichlet displacement; bottom)
	// 3: /       (Neumann condition;         top)

   ////////////////
   // displacement
   //
   {
	   // apply the spatial Dirichlet boundary conditions at each temporal DoF
	   BoundaryValues<dim> boundary_func;
	   
	   // remove old temporal support points
	   displacement_time_support_points.clear();
	   
	   FEValues<1> time_fe_values(
			slab->displacement_time_fe, 
			Quadrature<1>(slab->displacement_time_fe.get_unit_support_points()),
			update_quadrature_points
		);
	   std::vector<types::global_dof_index> time_local_dof_indices(slab->displacement_time_fe.n_dofs_per_cell());

	   for (const auto &time_cell : slab->displacement_time_dof_handler.active_cell_iterators())
	   {
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			// using temporal support points as quadrature points
			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				boundary_func.set_time(t_qq);
				displacement_time_support_points.insert(std::make_pair(t_qq, time_local_dof_indices[qq]));

				// determine spatial boundary values at temporal support point
				std::map<types::global_dof_index, double> boundary_values;
				// 0: u_x = 0 (Dirichlet displacement;   left)
				VectorTools::interpolate_boundary_values(space_dof_handler, 0, boundary_func, boundary_values, ComponentMask({true, false, false}));
				// 2: u_y = 0 (Dirichlet displacement; bottom)
				VectorTools::interpolate_boundary_values(space_dof_handler, 2, boundary_func, boundary_values, ComponentMask({false, true, false}));
				
				// calculate the correct space-time entry and apply the Dirichlet BC
				for (auto &entry : boundary_values)
				{
					types::global_dof_index id = entry.first + time_local_dof_indices[qq] * displacement_n_space_dofs;

					// apply BC
					for (typename SparseMatrix<double>::iterator p = system_matrix.begin(id); p != system_matrix.end(id); ++p)
						p->value() = 0.;
					system_matrix.set(id, id, 1.);
					system_rhs(id) = entry.second;
				}
			}
	   }
   }

   ////////////
   // pressure
   //
   {
	   // apply the spatial Dirichlet boundary conditions at each temporal DoF
	   BoundaryValues<dim> boundary_func;

	   // remove old temporal support points
	   pressure_time_support_points.clear();

	   FEValues<1> time_fe_values(
			slab->pressure_time_fe, 
			Quadrature<1>(slab->pressure_time_fe.get_unit_support_points()), 
			update_quadrature_points
		);
	   std::vector<types::global_dof_index> time_local_dof_indices(slab->pressure_time_fe.n_dofs_per_cell());

	   for (const auto &time_cell : slab->pressure_time_dof_handler.active_cell_iterators())
	   {
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			// using temporal support points as quadrature points
			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				boundary_func.set_time(t_qq);
				pressure_time_support_points.insert(std::make_pair(t_qq, time_local_dof_indices[qq]));

				// determine spatial boundary values at temporal support point
				std::map<types::global_dof_index, double> boundary_values;
				// 1:   p = 0 (Dirichlet pressure;      right)
				VectorTools::interpolate_boundary_values(space_dof_handler, 1, boundary_func, boundary_values, ComponentMask({false, false, true}));

				// calculate the correct space-time entry and apply the Dirichlet BC
				for (auto &entry : boundary_values)
				{
					// (NOTE: remember that pressure block is offset by number of displacement space-time DoFs)
					types::global_dof_index id = (entry.first-displacement_n_space_dofs) + time_local_dof_indices[qq] * pressure_n_space_dofs + displacement_n_dofs;

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
		std::max(slab->displacement_time_dof_handler.n_dofs(), slab->pressure_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
		Vector<double>(space_dof_handler.n_dofs()) // joint solution at timepoint t_qq
	);
	std::vector<double> displacement_values_t_qq;
	std::vector<double> pressure_values_t_qq;

	// fill the solutions on finer temporal mesh
	get_solution_on_finer_mesh(slab, solution_at_t_qq, displacement_values_t_qq, pressure_values_t_qq);
	
	bool displacement_is_finer = (slab->displacement_time_dof_handler.n_dofs() > slab->pressure_time_dof_handler.n_dofs());
	auto time_fe = (
		displacement_is_finer ?
			&(slab->displacement_time_fe) :
			&(slab->pressure_time_fe)
	);
	auto time_dof_handler = (
		displacement_is_finer ?
			&(slab->displacement_time_dof_handler) :
			&(slab->pressure_time_dof_handler)
	);
	FEValues<1> time_fe_values(*time_fe, QGauss<1>(time_fe->degree+4), update_values | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> time_local_dof_indices(time_fe->n_dofs_per_cell());
	
	QGauss<dim> space_quad_formula(space_fe.degree+8);
	QGauss<dim-1> space_face_quad_formula(space_fe.degree+8);
	
	FEValues<dim> space_fe_values(space_fe, space_quad_formula,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> space_fe_face_values(space_fe, space_face_quad_formula,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
	std::vector<types::global_dof_index> space_local_dof_indices(space_fe.n_dofs_per_cell());

	for (const auto &space_cell : space_dof_handler.active_cell_iterators()) {
		space_fe_values.reinit(space_cell);
		space_cell->get_dof_indices(space_local_dof_indices);

		for (const unsigned int space_face : space_cell->face_indices())
		if (space_cell->at_boundary(space_face) && (space_cell->face(space_face)->boundary_id() == 2)) // face is at bottom boundary
		{
			space_fe_face_values.reinit(space_cell, space_face);

			for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
				time_fe_values.reinit(time_cell);
				time_cell->get_dof_indices(time_local_dof_indices);

				for (const unsigned int q : space_fe_face_values.quadrature_point_indices())
				for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
				{
					double solution_p_x_q = 0.;
					for (const unsigned int j : space_fe_face_values.dof_indices())
					for (const unsigned int jj : time_fe_values.dof_indices())
					{
						solution_p_x_q += solution_at_t_qq[time_local_dof_indices[jj]](space_local_dof_indices[j]) * space_fe_values[pressure].value(j, q) * time_fe_values.shape_value(jj, qq);
					}

					// add local contributions to global pressure over bottom boundary
					goal_func_value += solution_p_x_q * space_fe_face_values.JxW(q) * time_fe_values.JxW(qq);
				}
			}
		}
	}
}

template<int dim>
void SpaceTime<dim>::process_solution(std::shared_ptr<Slab> &slab, const unsigned int cycle, bool last_slab) {
	// TODO: remove hp::QCollection from here
	
// 	std::vector<Vector<double>> solution_at_t_qq(
// 		std::max(slab->displacement_time_dof_handler.n_dofs(), slab->pressure_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
// 		Vector<double>(space_dof_handler.n_dofs()) // joint solution at timepoint t_qq
// 	);
// 	std::vector<double> displacement_values_t_qq;
// 	std::vector<double> pressure_values_t_qq;

// 	// fill the solutions on finer temporal mesh
// 	get_solution_on_finer_mesh(slab, solution_at_t_qq, displacement_values_t_qq, pressure_values_t_qq);
	
// 	bool displacement_is_finer = (slab->displacement_time_dof_handler.n_dofs() > slab->pressure_time_dof_handler.n_dofs());
// 	auto time_fe = (
// 		displacement_is_finer ?
// 			&(slab->displacement_time_fe) :
// 			&(slab->pressure_time_fe)
// 	);
// 	auto time_dof_handler = (
// 		displacement_is_finer ?
// 			&(slab->displacement_time_dof_handler) :
// 			&(slab->pressure_time_dof_handler)
// 	);
// 	FEValues<1> time_fe_values(*time_fe, QGauss<1>(time_fe->degree+4), update_values | update_quadrature_points | update_JxW_values);
// 	std::vector<types::global_dof_index> time_local_dof_indices(time_fe->n_dofs_per_cell());
	
// 	QGauss<dim> displacement_space_quad_formula(displacement_space_fe.degree+8);
// 	QGauss<dim> pressure_space_quad_formula(pressure_space_fe.degree+8);
	
// 	hp::QCollection<dim> space_q_collection;
// 	space_q_collection.push_back(displacement_space_quad_formula);
// 	space_q_collection.push_back(pressure_space_quad_formula);

// 	hp::FEValues<dim> hp_space_fe_values(space_fe_collection, space_q_collection,
// 			update_values | update_gradients | update_quadrature_points | update_JxW_values);
// 	Solution<dim> solution_func;

// 	DynamicSparsityPattern space_dsp(space_dof_handler.n_dofs(), space_dof_handler.n_dofs());
// 	DoFTools::make_flux_sparsity_pattern(space_dof_handler, space_dsp);
// 	SparsityPattern sparsity;
// 	sparsity.copy_from(space_dsp);
// 	SparseMatrix<double> space_mass_matrix(sparsity);
// 	MatrixCreator::create_mass_matrix(space_dof_handler, space_q_collection, space_mass_matrix);
	
// 	for (const auto &time_cell : time_dof_handler->active_cell_iterators()) {
// 	  time_fe_values.reinit(time_cell);
// 	  time_cell->get_dof_indices(time_local_dof_indices);

// 	  for (const unsigned int qq : time_fe_values.quadrature_point_indices()) 
// 	  {
// 	    // time quadrature point
// 	    double t_qq = time_fe_values.quadrature_point(qq)[0];
// 	    solution_func.set_time(t_qq);
	    
// 	    // get the space solution at the quadrature point
// 	    Vector<double> space_solution(space_dof_handler.n_dofs());
// 	    for (const unsigned int ii : time_fe_values.dof_indices())
// 	    {
// 	      for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
// 			space_solution(i) += solution_at_t_qq[time_local_dof_indices[ii]](i) * time_fe_values.shape_value(ii, qq);
// 	    }
	     	      
// 	    // compute the L2 error at the temporal quadrature point
// 		Vector<double> analytical_solution(space_dof_handler.n_dofs());
// 		VectorTools::interpolate(space_dof_handler,
// 					solution_func,
// 					analytical_solution,
// 					ComponentMask());
			
// 		// e = u - u_h
// 		analytical_solution.add(-1., space_solution);

// 	    // compute_global_error by hand
// 		//     error(t_qq) = e * M * e
// 		Vector<double> tmp(space_dof_handler.n_dofs());
// 		space_mass_matrix.vmult(tmp, analytical_solution);
// 		double L2_error_t_qq = (tmp * analytical_solution);
	    
// 	    // add local contributions to global L2 error
// 	    L2_error +=  L2_error_t_qq * time_fe_values.JxW(qq);
// 	  }
// 	}

// //	n_active_time_cells += slab->time_triangulation.n_active_cells();
// //	n_time_dofs += (slab->time_dof_handler.n_dofs()-1); // first time DoF is also part of last slab

// 	displacement_total_n_dofs += displacement_n_dofs;
// 	pressure_total_n_dofs += pressure_n_dofs;

// 	if (last_slab)
// 	{
// 		L2_error = std::sqrt(L2_error);
// 		L2_error_vals.push_back(L2_error);

// 		// add values to
// //		const unsigned int n_active_cells = space_triangulation.n_active_cells() * n_active_time_cells;
// //		const unsigned int n_space_dofs   = space_dof_handler.n_dofs();
// //		const unsigned int n_dofs         = n_space_dofs * n_time_dofs;
// 		const unsigned int n_dofs = displacement_total_n_dofs + pressure_total_n_dofs;

// 		convergence_table.add_value("cycle", cycle);
// //		convergence_table.add_value("cells", n_active_cells);
// 		convergence_table.add_value("dofs", n_dofs);
// //		convergence_table.add_value("dofs(space)", n_space_dofs);
// //		convergence_table.add_value("dofs(time)", n_time_dofs);
// 		convergence_table.add_value("L2", L2_error);
// 	}
}

template<int dim>
void SpaceTime<dim>::print_convergence_table() {
	convergence_table.set_precision("L2", 3);
	convergence_table.set_scientific("L2", true);
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
void SpaceTime<dim>::get_solution_on_finer_mesh(std::shared_ptr<Slab> &slab, std::vector<Vector<double>> &solution_at_t_qq, std::vector<double> &displacement_values_t_qq, std::vector<double> &pressure_values_t_qq) {
	// NOTE: For dG(0) it is very simply to get the solution on the finer mesh, since on just needs to store duplicates of the solution on the coarser temporal mesh 
	
	//////////
	// displacement
	//
	{
		// Iterated formula splits temporal element into many subintervals
		// Should evaluate displacement and pressure at same temporal points to compare in Paraview
		int num_splits = (initial_temporal_ref_displacement >= initial_temporal_ref_pressure) ? 1 : std::pow(2, initial_temporal_ref_pressure);
		QIterated<1> quad(QTrapez<1>(), num_splits);
		// std::cout << "quadrature formula size: " << quad.size() << std::endl;
		// custom quadrature which skips the first quadrature point from the trapezoidal rule
		std::vector< Point<1> > quad_points(quad.size()-1);
		for (unsigned int i = 0; i < quad.size()-1; ++i)
			quad_points[i] = quad.point(i+1);

		FEValues<1> time_fe_values(slab->displacement_time_fe, Quadrature<1> (quad_points), update_values | update_quadrature_points | update_JxW_values);
		std::vector<types::global_dof_index> time_local_dof_indices(slab->displacement_time_fe.n_dofs_per_cell());

		unsigned int n_local_snapshots = 0;

		bool is_first_time_cell = true;

		for (const auto &time_cell : slab->displacement_time_dof_handler.active_cell_iterators())
		{
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				displacement_values_t_qq.push_back(t_qq);

				// get the FEM space solution at the quadrature point
				for (const unsigned int ii : time_fe_values.dof_indices())
				{
					for (unsigned int i = 0; i < displacement_n_space_dofs; ++i)
						solution_at_t_qq[n_local_snapshots](i) += solution(i + time_local_dof_indices[ii] * displacement_n_space_dofs + 0) * time_fe_values.shape_value(ii, qq);
				}

				n_local_snapshots++;
			}

			is_first_time_cell = false;
		}
	}

	Assert(solution_at_t_qq.size() == displacement_values_t_qq.size(),
		ExcDimensionMismatch(solution_at_t_qq.size(), displacement_values_t_qq.size()));

	//////////
	// pressure
	//
	{
		// Iterated formula splits temporal element into many subintervals
		// Should evaluate displacement and pressure at same temporal points to compare in Paraview
		int num_splits = (initial_temporal_ref_displacement > initial_temporal_ref_pressure) ? std::pow(2, initial_temporal_ref_displacement) : 1;
		QIterated<1> quad(QTrapez<1>(), num_splits);
		// std::cout << "quadrature formula size: " << quad.size() << std::endl;
		// custom quadrature which skips the first quadrature point from the trapezoidal rule
		std::vector< Point<1> > quad_points(quad.size()-1);
		for (unsigned int i = 0; i < quad.size()-1; ++i)
			quad_points[i] = quad.point(i+1);

		FEValues<1> time_fe_values(slab->pressure_time_fe, Quadrature<1>(quad_points), update_values | update_quadrature_points | update_JxW_values);
		std::vector<types::global_dof_index> time_local_dof_indices(slab->pressure_time_fe.n_dofs_per_cell());

		unsigned int n_local_snapshots = 0;

		for (const auto &time_cell : slab->pressure_time_dof_handler.active_cell_iterators()) {
			time_fe_values.reinit(time_cell);
			time_cell->get_dof_indices(time_local_dof_indices);

			for (const unsigned int qq : time_fe_values.quadrature_point_indices())
			{
				// time quadrature point
				double t_qq = time_fe_values.quadrature_point(qq)[0];
				pressure_values_t_qq.push_back(t_qq);

				// get the FEM space solution at the quadrature point
				for (const unsigned int ii : time_fe_values.dof_indices())
				{
					for (unsigned int i = 0; i < pressure_n_space_dofs; ++i)
						solution_at_t_qq[n_local_snapshots](i + displacement_n_space_dofs) += solution(i + time_local_dof_indices[ii] * pressure_n_space_dofs + displacement_n_dofs) * time_fe_values.shape_value(ii, qq);
				}

				n_local_snapshots++;
			}
		}
	}

	Assert(solution_at_t_qq.size() == pressure_values_t_qq.size(),
		ExcDimensionMismatch(solution_at_t_qq.size(), pressure_values_t_qq.size()));

	for (unsigned int i = 0; i < solution_at_t_qq.size(); ++i)
		Assert(displacement_values_t_qq[i] == pressure_values_t_qq[i], ExcNotImplemented());
}

template<>
void SpaceTime<2>::output_results(
        std::shared_ptr<Slab> &slab, const unsigned int refinement_cycle, unsigned int slab_number, bool last_slab) {
        std::string output_dir = "output/dim=2/cycle=" + std::to_string(refinement_cycle) + "/";

        // output results as VTK files
		std::vector<Vector<double>> solution_at_t_qq(
			std::max(slab->displacement_time_dof_handler.n_dofs(), slab->pressure_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
			Vector<double>(space_dof_handler.n_dofs()) // joint solution at timepoint t_qq
		);
		std::vector<double> displacement_values_t_qq;
		std::vector<double> pressure_values_t_qq;

		// fill the solutions on finer temporal mesh
		get_solution_on_finer_mesh(slab, solution_at_t_qq, displacement_values_t_qq, pressure_values_t_qq);		

		// output the solution as vtk
		for (unsigned int i = 0; i < solution_at_t_qq.size(); i++)
		{
			DataOut<2> data_out;
			data_out.attach_dof_handler(space_dof_handler);

			std::vector<std::string> solution_names;
			solution_names.push_back("x_displacement");
			solution_names.push_back("y_displacement");
			solution_names.push_back("pressure");

			std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(2+1, DataComponentInterpretation::component_is_scalar);
			data_out.attach_dof_handler(space_dof_handler);  
			data_out.add_data_vector(solution_at_t_qq[i], solution_names, DataOut<2>::type_dof_data, data_component_interpretation);
			data_out.build_patches(0);

			data_out.set_flags(DataOutBase::VtkFlags(displacement_values_t_qq[i], n_snapshots));
			// std::cout << "t = " << displacement_values_t_qq[i] << " ;i = " << n_snapshots << std::endl;

			std::ofstream output(output_dir + "solution_" + Utilities::int_to_string(n_snapshots, 5) + ".vtk");
			data_out.write_vtk(output);

			n_snapshots++;
		}
}

template <int dim>
void SpaceTime<dim>::compute_functional_values(std::shared_ptr<Slab> &slab,
                                               const unsigned int cycle,
                                               bool first_slab) {
  std::string output_dir = "output/dim=" + std::to_string(dim) + "/cycle=" + std::to_string(cycle) + "/";

  if (first_slab) {
	// space
	std::vector<Point<dim>> space_support_points(space_dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(MappingQ1<dim, dim>(), space_dof_handler,
										space_support_points);

	pressure_dofmap.clear();
	x_disp_dofmap.clear();

	unsigned int i = 0;
	for (auto point : space_support_points) {
		// fill DoF maps for bottom boundary
		if (point[1] == 0.)  // y == 0
		{
		if (i >= displacement_n_space_dofs) {
			// pressure DoF
			pressure_dofmap.insert({point[0], i});
		} else {
			// displacement DoF
			if (i % 2 == 0) {
			// x-displacement DoF
			x_disp_dofmap.insert({point[0], i});
			}
		}
		}

		++i;
	}

#if DEBUG
    std::cout << "Pressure BC DoFs:" << std::endl;
    for (auto &entry : pressure_dofmap)
      std::cout << entry.first << " " << entry.second << std::endl;

    std::cout << "\nX-Displacement BC DoFs:" << std::endl;
    for (auto &entry : x_disp_dofmap)
      std::cout << entry.first << " " << entry.second << std::endl;
#endif

    std::ofstream pressure_out(output_dir + "pressure.txt");
    std::ofstream x_disp_out(output_dir + "x_displacement.txt");

    pressure_out << "Coordinates:" << std::endl;
    for (auto &entry : pressure_dofmap) pressure_out << " " << entry.first;
    pressure_out << std::endl;

    x_disp_out << "Coordinates:" << std::endl;
    for (auto &entry : x_disp_dofmap) x_disp_out << " " << entry.first;
    x_disp_out << std::endl;

    pressure_out << "\nTime | List of values" << std::endl;
    x_disp_out << "\nTime | List of values" << std::endl;

    pressure_out.close();
    x_disp_out.close();
  }

  std::ofstream pressure_out(output_dir + "pressure.txt", std::ios::app);
  std::ofstream x_disp_out(output_dir + "x_displacement.txt", std::ios::app);

  std::vector<Vector<double>> solution_at_t_qq(
	  std::max(slab->displacement_time_dof_handler.n_dofs(), slab->pressure_time_dof_handler.n_dofs()), // number of time DoFs on finer mesh
	  Vector<double>(space_dof_handler.n_dofs())														// joint solution at timepoint t_qq
  );
  std::vector<double> displacement_values_t_qq;
  std::vector<double> pressure_values_t_qq;

  // fill the solutions on finer temporal mesh
  get_solution_on_finer_mesh(slab, solution_at_t_qq, displacement_values_t_qq, pressure_values_t_qq);

  bool displacement_is_finer = (slab->displacement_time_dof_handler.n_dofs() > slab->pressure_time_dof_handler.n_dofs());
  auto time_fe = (displacement_is_finer ? &(slab->displacement_time_fe) : &(slab->pressure_time_fe));
  auto time_dof_handler = (displacement_is_finer ? &(slab->displacement_time_dof_handler) : &(slab->pressure_time_dof_handler));
  FEValues<1> time_fe_values(*time_fe, Quadrature<1>({Point<1>(1.)}), update_values | update_quadrature_points | update_JxW_values);
  std::vector<types::global_dof_index> time_local_dof_indices(time_fe->n_dofs_per_cell());

  for (const auto &time_cell : time_dof_handler->active_cell_iterators())
  {
	time_fe_values.reinit(time_cell);
	time_cell->get_dof_indices(time_local_dof_indices);

	for (const unsigned int qq : time_fe_values.quadrature_point_indices())
	{
	  // time quadrature point
	  double t_qq = time_fe_values.quadrature_point(qq)[0];

	  // get the FEM space solution at the quadrature point
	  Vector<double> space_solution(space_dof_handler.n_dofs());
	  for (const unsigned int ii : time_fe_values.dof_indices())
	  {
		for (unsigned int i = 0; i < space_dof_handler.n_dofs(); ++i)
			space_solution(i) += solution_at_t_qq[time_local_dof_indices[ii]](i) *
								 time_fe_values.shape_value(ii, qq);
	  }

	  pressure_out << t_qq << " |";
	  x_disp_out << t_qq << " |";

	  // print pressure and x-displacement at bottom boundary
	  for (auto &entry : pressure_dofmap)
		pressure_out << " " << space_solution[entry.second];
	  pressure_out << std::endl;

	  for (auto &entry : x_disp_dofmap)
		x_disp_out << " " << space_solution[entry.second];
	  x_disp_out << std::endl;
	}
  }
}

template<int dim>
void SpaceTime<dim>::run() {
	Assert (dim==2, ExcInternalError());

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
		displacement_total_n_dofs = 0;
		pressure_total_n_dofs = 0;
		L2_error = 0.;
		goal_func_value = 0.;
		n_snapshots = 0;

		// create output directory if necessary
        std::string dim_dir = "output/dim=" + std::to_string(dim) + "/";
        std::string output_dir = dim_dir + "cycle=" + std::to_string(cycle) + "/";
        for (auto dir : { "output/", dim_dir.c_str(), output_dir.c_str() })
            mkdir(dir, S_IRWXU);

		////////////////////////////////////////////
		// create spatial DoFHandler
		//
		space_dof_handler.distribute_dofs(space_fe);
		
		// Renumber spatials DoFs into displacement, pressure
  		DoFRenumbering::component_wise(space_dof_handler, {0, 0, 1});

		// number of space DoFs per component
		const std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(space_dof_handler, {0, 0, 1});
		displacement_n_space_dofs = dofs_per_block[0];
		pressure_n_space_dofs = dofs_per_block[1];

		////////////////////////////////////////////
		// initial value: u(0) and v(0)
		//

		// compute displacement and pressure initial value vector
		Vector<double> initial_solution(space_dof_handler.n_dofs());
		VectorTools::interpolate(
			space_dof_handler,
			InitialValues<dim>(),
			initial_solution,
			ComponentMask()
		);

		// fill initial solution for displacement and pressure
		initial_solution_displacement.reinit(space_dof_handler.n_dofs());
		initial_solution_pressure.reinit(space_dof_handler.n_dofs());
		// displacement
		for (unsigned int i = 0; i < displacement_n_space_dofs; ++i)
			initial_solution_displacement(i) = initial_solution(i);
		// pressure
		for (unsigned int i = 0; i < pressure_n_space_dofs; ++i)
			initial_solution_pressure(i + displacement_n_space_dofs) = initial_solution(i + displacement_n_space_dofs);

		/////////////////////////////
		//    TIME-SLABBING LOOP
		//
		for (unsigned int k = 0; k < slabs.size(); ++k) {
			// create and solve linear system
			setup_system(slabs[k], k+1);		
			assemble_system(slabs[k], k==0);
			solve(k == 0);

#if DEBUG
			std::cout << "solution.linfty_norm() = " << solution.linfty_norm() << std::endl;
			std::ofstream solution_out("solution_" + Utilities::int_to_string(k, 5) + ".txt");
	    	solution.print(solution_out, /*precision*/16);
#endif

			// write system matrix out to file on the first slab to compute the condition number
#if CONDITION
			if (k == 0)
			{
				std::ofstream matrix_out(output_dir + "matrix.txt");
				print_as_numpy_arrays_high_resolution(system_matrix, matrix_out, /*precision*/16);
			}
#endif

#if VTK_OUTPUT
			// output vtk
			output_results(slabs[k], cycle, k, (k == slabs.size()-1));
#endif

// 			// Compute the error to the zero function (no analytical solution)
// 			process_solution(slabs[k], cycle, (k == slabs.size()-1));

#if DIM == 2
			// compute goal functional
			compute_goal_functional(slabs[k]);
#endif

			// Compute goal functionals for some time points
      		compute_functional_values(slabs[k], cycle, k == 0);

			///////////////////////
			// prepare next slab
			//

			// NOTE: this getting of initial values only works for QGaussLobatto in time, when the last temporal index corresponds to the last temporal quadrature point on the temporal element 
			// get initial value for next slab
			// displacement
			for (unsigned int i = 0; i < displacement_n_space_dofs; ++i)
				initial_solution_displacement(i) = solution(i + displacement_n_dofs - displacement_n_space_dofs);
			// pressure
			for (unsigned int i = 0; i < pressure_n_space_dofs; ++i)
				initial_solution_pressure(i + displacement_n_space_dofs) = solution(i + pressure_n_dofs - pressure_n_space_dofs + displacement_n_dofs);
		}

		// create plots for time evolution of functional values
		int _n_time_elements_displacement = slabs[0]->displacement_time_triangulation.n_active_cells();
		int _n_time_elements_pressure = slabs[0]->pressure_time_triangulation.n_active_cells();
		std::string plotting_cmd = "python3 plot_functional_values.py --name 'dG(" 
								+ std::to_string(slabs[0]->displacement_time_fe.get_degree()) 
								+ ") with ($\\mathbb{T}_k^u$:$\\mathbb{T}_k^p$) = "
								+ std::to_string(_n_time_elements_displacement)
								+ ":"
								+ std::to_string(_n_time_elements_pressure)
								+ "'";
		// std::cout << plotting_cmd << std::endl;
		system(plotting_cmd.c_str());

		goal_func_vals.push_back(goal_func_value);

		// refine mesh
		if (cycle < max_n_refinement_cycles - 1) {
			space_triangulation.refine_global(refine_space);
			
			if (split_slabs) {
				std::vector<std::shared_ptr<Slab> > split_slabs;
				for (auto &slab : slabs) {
					// NOTE: using same temporal degree for displacement and pressure
					Assert(slab->displacement_time_fe.get_degree() == slab->pressure_time_fe.get_degree(), ExcNotImplemented());
					split_slabs.push_back(
						std::make_shared<Slab>(
							slab->displacement_time_fe.get_degree(),
							slab->start_time,
							0.5 * (slab->start_time + slab->end_time)
						)
					);
					split_slabs.push_back(
						std::make_shared<Slab>(
							slab->displacement_time_fe.get_degree(),
							0.5 * (slab->start_time + slab->end_time),
							slab->end_time
						)
					);
				}
				slabs = split_slabs;

				for (auto &slab : slabs) {
					GridGenerator::hyper_rectangle(
						slab->displacement_time_triangulation,
						Point<1>(slab->start_time),
						Point<1>(slab->end_time)
					);
					slab->displacement_time_triangulation.refine_global(initial_temporal_ref_displacement);
					GridGenerator::hyper_rectangle(
						slab->pressure_time_triangulation,
						Point<1>(slab->start_time),
						Point<1>(slab->end_time)
					);
					slab->pressure_time_triangulation.refine_global(initial_temporal_ref_pressure);
				}
			} else {
				for (auto &slab : slabs) {
					slab->displacement_time_triangulation.refine_global(refine_time);
					slab->pressure_time_triangulation.refine_global(refine_time);
				}
			}
		}
	}

	// print_convergence_table();

#if DIM == 2
	std::cout << "Goal functional values:" << std::endl;
	for (unsigned int cycle = 0; cycle < max_n_refinement_cycles; ++cycle) {
		std::cout.precision(15);
		std::cout.setf(std::ios::scientific, std::ios::floatfield);
		std::cout << "Cycle " << cycle << ": " << goal_func_vals[cycle] << std::endl;
	}

	double reference_value = 8.718830861443711e+13; // N = 500,000; pressure_ref = 0; displacement_ref = 0
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

		// use finer temporal displacement grid as default
		unsigned int pressure_ref = 0;
		unsigned int displacement_ref = 0;
		
		// parse command line arguments
		if (argc > 1)
		{
			for (int i = 1; i < argc; ++i)
			{
				if (std::string(argv[i]) == std::string("-pressure_ref"))
				{
					pressure_ref = std::stoi(argv[i+1]);
					i++;
				}
				else if (std::string(argv[i]) == std::string("-displacement_ref"))
				{
					displacement_ref = std::stoi(argv[i+1]);
					i++;
				}
			}
		}
		std::cout << "CLI ARGUMENTS:" << std::endl;
		std::cout << "displacement_ref = " << displacement_ref << std::endl;
		std::cout << "pressure_ref = " << pressure_ref << std::endl;

		// run the simulation
#if DIM == 2
		// 2+1D:
		// -----
		std::vector<unsigned int> r;
		std::vector<double> t = { 0. };
		double T = 5.0e+6;
		int N = 1250;
		// int N = 500000; // reference calculation
		double dt = T / N;
		std::cout << "T = " << T << std::endl;
		std::cout << "N = " << N << std::endl;
		std::cout << "dt = " << dt << std::endl;
		for (unsigned int i = 0; i < N; ++i) {
			r.push_back(0); 
			t.push_back((i + 1) * dt);
		} 
		SpaceTime<2> space_time_problem(
			2,                      // s_displacement ->  spatial FE degree (u)
			1,						// s_pressure     ->  spatial FE degree (p)
			r,         				// r -> temporal FE degree
			t, 						// time points
			4,                      // max_n_refinement_cycles,
			displacement_ref,       // initial_temporal_ref_displacement
			pressure_ref,           // initial_temporal_ref_pressure
			false,                  // refine_space
			true,                   // refine_time
			true                    // split_slabs
		);
#endif

		// run the simulation
		space_time_problem.run();

		// save final grid
		space_time_problem.print_grids(
			"space_grid.svg", 
			"time_grid_displacement.svg", 
			"time_grid_pressure.svg", 
			"time_grid_joint.svg"
		);
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
