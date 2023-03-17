
			===========================================
			MEDIQA-Chat 2023 Test Sets (March 15, 2023)
			===========================================

===========================================
Test Sets:
===========================================

1) Task A test set consists of 200 doctor-patient conversations to summarize and generate associated section headers and contents.

The full list of normalized section headers: 

	1. fam/sochx 
	2. genhx 
	3. pastmedicalhx 
	4. cc 
	5. pastsurgical 
	6. allergy
	7. ros
	8. medications
	9. assessment
	10. exam
	11. diagnosis
	12. disposition
	13. plan
	14. edcourse 
	15. immunizations
	16. imaging
	17. gynhx 
	18. procedures
	19. other_history
	20. labs


2) Task B test set consists of 40 full doctor-patient conversations to summarize and generate a full encounter note with four main sections. Accepted first-level section headers are: "HISTORY OF PRESENT ILLNESS", "PHYSICAL EXAM", "RESULTS", "ASSESSMENT AND PLAN".


Full encounter notes are expected to have at least one of four overall section divisions demarked by the first-occuring of its related section headers :

	| note_division | section_headers
	---------------------------------------------------------------------------
	| subjective | chief complaint, history of present illness, hpi, subjective
	---------------------------------------------------------------------------
	| objective_exam | physical exam, exam
	---------------------------------------------------------------------------
	| objective_results | results, findings
	---------------------------------------------------------------------------
	| assessment_and_plan | assessment, plan

Depending on the encounter, objective_exam and objective_results may not be relevant.
We encourage review the sample data as well as the evaluation script to understand the best demarkation headers for your generated note.
	


3) Task C test set consists of 40 full notes to use for the generation of relevant doctor-patient conversations. 


===========================================
Run submission forms: 
===========================================

Task A: https://forms.gle/7zuRwSEFVpMjEYBh9 
Task B: https://forms.gle/1W2atpoHr1jCT46F8 
Task C: https://forms.gle/jDDBvMW9UiwnfRsF8 

===========================================
Code Submission: 
===========================================

We provided guidelines for code preparation/submission here: https://github.com/abachaa/MEDIQA-Chat-2023/blob/main/README.md 

Each team should create a private GitHub repo with the team’s code and add the following users: abachaa, wyim, griff4692.  

The GitHub repo should be named as follows: MEDIQA-Chat-2023-TeamName.  


	The deadline to submit the runs and codes is Friday March 17.  

===========================================
Evaluations: 
===========================================

We will use an ensemble metric that combines ROUGE-1 F1, BERTScore F1, and BLEURT (equally weighted) to evaluate and rank the submitted runs in the three tasks.  

Task C will have an additional downstream evaluation to assess the impact of training summarization models using task B training data augmented with the generated conversations. Task C will thus have 2 evaluation scores/rankings: (i) an intrinsic evaluation using reference doctor-patient conversations and (ii) an extrinsic evaluation of summarization models fine-tuned with the augmented dataset.  


	Official results will be released on March 31, 2023. 



Good luck!   

The MEDIQA-Chat organizers:   

Asma, Wen-wai, Griffin, Neal, and Meliha 

