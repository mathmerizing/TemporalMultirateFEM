echo ""
echo "**********************************************************"
echo "RUNNING 1+1D HEAT-WAVE CODE TO PRODUCE RESULTS FOR PAPER *"
echo "**********************************************************"

# check whether executable for 1+1D heat-wave exists
EXECUTABLE=main_1d
if [ -f "$EXECUTABLE" ]; then
	echo "   CHECK: The executable '$EXECUTABLE' exists."
else
	echo "   CHECK: The executable '$EXECUTABLE' does not exist."
	echo "   INFO:  You need to set '#   define DIM 1' in line 108 in main.cc."
	echo "          Then create an executable and rename it from 'main' to 'main_1d'."
	echo "ABORT"
	exit 1
fi

# remove old 1D output files
mkdir -p output/dim=1/
cd output/dim=1/
if [ "$(ls -A .)" ]; then
	rm -r cycle=*/
fi

if ! [ "$(ls -A .)" ]; then
	echo "   CHECK: Output folder is empty." 
else
	echo "   CHECK: Output folder is NOT empty."
	echo "ABORT"
	exit 1
fi
cd ../../

# create output directory if necessary
mkdir -p paper_output_1d/

#######################################################
# run results for different solid and fluid refinements
#
cd paper_output_1d/
# clear old results
if [ " -d refinement_analysis/" ]; then
	rm -r refinement_analysis/
fi
mkdir -p refinement_analysis/
cd ../

echo ""
echo "STARTING REFINEMENT ANALYSIS..."
start_time_refinement=`date +%s`
n_runs=1
coarse_ref=0
for fine_ref in {0..8}
do
	echo ""
	echo "   RUN #$n_runs: main_1d with solid_ref=$fine_ref and fluid_ref=$coarse_ref"
	echo "   --------------------------------------------------" 
	OUTPUT_DIR="paper_output_1d/refinement_analysis/solid=$fine_ref-fluid=$coarse_ref/"
	#echo $OUTPUT_DIR	
	mkdir -p $OUTPUT_DIR

	start_time=`date +%s`
	# run code
	./main_1d -solid_ref $fine_ref -fluid_ref $coarse_ref >> $OUTPUT_DIR/console_output.log
	end_time=`date +%s`
	echo "      TIME: Running the code took $((end_time-start_time)) seconds."
	mv output/dim=1/ $OUTPUT_DIR/output/
	echo "      INFO: Moved all output files."	
	echo "   DONE"
	((n_runs=n_runs+1))

	if [ "$fine_ref" -gt 0 ]; then
		echo ""
		echo "   RUN #$n_runs: main_1d with solid_ref=$coarse_ref and fluid_ref=$fine_ref"
		echo "   --------------------------------------------------" 
		OUTPUT_DIR="paper_output_1d/refinement_analysis/solid=$coarse_ref-fluid=$fine_ref/"
		#echo $OUTPUT_DIR	
		mkdir -p $OUTPUT_DIR

		start_time=`date +%s`
		# run code
		./main_1d -solid_ref $coarse_ref -fluid_ref $fine_ref >> $OUTPUT_DIR/console_output.log
		end_time=`date +%s`
		echo "      TIME: Running the code took $((end_time-start_time)) seconds."
		mv output/dim=1/ $OUTPUT_DIR/output/
		echo "      INFO: Moved all output files."	
		echo "   DONE"
		((n_runs=n_runs+1))
	fi
done
end_time_refinement=`date +%s`
echo "TIME: Running the refinement analysis took $((end_time_refinement-start_time_refinement)) seconds."
echo ""

#####################################################
# run results for different penalty parameters gamma
#

# TODO


echo "DONE"

echo ""
#echo "REMINDER: Set max_n_refinement_cycles to 5 for paper"
