SELECT student_id
FROM students
GROUP BY student_id
HAVING min(class_score) > 90;

