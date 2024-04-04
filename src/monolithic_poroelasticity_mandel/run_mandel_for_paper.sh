echo ""
echo "**************************************************"
echo "RUNNING MANDEL CODE TO PRODUCE RESULTS FOR PAPER *"
echo "**************************************************"

# check whether executable exists
EXECUTABLE=main
if [ -f "$EXECUTABLE" ]; then
	echo "   CHECK: The executable '$EXECUTABLE' exists."
else
	echo "   CHECK: The executable '$EXECUTABLE' does not exist."
	echo "ABORT"
	exit 1
fi

# remove old 2D output files
mkdir -p output/dim=2/
cd output/dim=2/
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
mkdir -p paper_output/

##################################################################
# run results for different displacememt and pressure refinements
#
cd paper_output/
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
for fine_ref in {0..4}
do
	echo ""
	echo "   RUN #$n_runs: main with displacement_ref=$fine_ref and pressure_ref=$coarse_ref"
	echo "   --------------------------------------------------" 
	OUTPUT_DIR="paper_output/refinement_analysis/displacement=$fine_ref-pressure=$coarse_ref/"
	#echo $OUTPUT_DIR	
	mkdir -p $OUTPUT_DIR

	start_time=`date +%s`
	# run code
	./main -displacement_ref $fine_ref -pressure_ref $coarse_ref >> $OUTPUT_DIR/console_output.log
	end_time=`date +%s`
	echo "      TIME: Running the code took $((end_time-start_time)) seconds."
	mv output/dim=2/ $OUTPUT_DIR/output/
	echo "      INFO: Moved all output files."	
	echo "   DONE"
	((n_runs=n_runs+1))

	if [ "$fine_ref" -gt 0 ]; then
		echo ""
		echo "   RUN #$n_runs: main with displacement_ref=$coarse_ref and pressure_ref=$fine_ref"
		echo "   --------------------------------------------------" 
		OUTPUT_DIR="paper_output/refinement_analysis/displacement=$coarse_ref-pressure=$fine_ref/"
		#echo $OUTPUT_DIR	
		mkdir -p $OUTPUT_DIR

		start_time=`date +%s`
		# run code
		./main -displacement_ref $coarse_ref -pressure_ref $fine_ref >> $OUTPUT_DIR/console_output.log
		end_time=`date +%s`
		echo "      TIME: Running the code took $((end_time-start_time)) seconds."
		mv output/dim=2/ $OUTPUT_DIR/output/
		echo "      INFO: Moved all output files."	
		echo "   DONE"
		((n_runs=n_runs+1))
	fi
done
end_time_refinement=`date +%s`
echo "TIME: Running the refinement analysis took $((end_time_refinement-start_time_refinement)) seconds."
echo ""


echo "DONE"

echo ""
