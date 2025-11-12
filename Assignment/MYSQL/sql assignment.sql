create database MarketCo;
use MarketCo;
create table company (
companyid int  primary key ,
companyname varchar(50),
street varchar(50),
city varchar(50),
state varchar(20),
zip varchar(10)
);




 /*
 1) Statement to create the Contact table
 */

create table contact(
contactid int primary key ,
companyid int,
foreign key contact(companyid) references company(companyid),
firstname varchar(50),
lastname varchar(50),
street varchar(50),
city varchar(50),
state varchar(10),
zip varchar(10),
ismain boolean,
email varchar(45),
phone varchar(12)
);

/*
3) Statement to create the ContactEmployee table
HINT: Use DATE as the datatype for ContactDate. It allows you to store the
date in this format: YYYY-MM-DD (i.e., ‘2014-03-12’ for March 12, 2014).
*/

create table  contactemployee (
contactemployeeid int primary key,
contactID int,
 foreign key contactemployee(contactID ) references contact(contactid),
 emplpoyeeid int,
 contactDate date,
 Description varchar(100)
 );

/* 
2) Statement to create the Employee table
*/



 
 create table employee(
 employeeid int primary key,
 firstname varchar(50),
 lastname varchar(50),
 salary decimal(10,2),
 hiredate date,
 jobtitle varchar(25),
 email varchar(45),
 phone varchar(12)
 );
 
 
 alter table contactemployee
 add foreign key (emplpoyeeidq) references employee(employeeid);
 

/*
4) In the Employee table, the statement that changes Lesley Bland’s phone number
to 215-555-8800
*/

update employee  set phone = "215-555-8800" where  firstname =" lesley"  and lastname ="bland" ;

 
 /*
 5) In the Company table, the statement that changes the name of “Urban
Outfitters, Inc.” to “Urban Outfitters”
*/
update  company  set companyname = "urban outfiter" where companyname =  "urban outfiter inc" ;
 
 
/* 6) In ContactEmployee table, the statement that removes Dianne Connor’s contact
event with Jack Lee (one statement).
HINT: Use the primary key of the ContactEmployee table to specify the correct record to remove
*/



 DELETE FROM ContactEmployee
WHERE ContactID = (SELECT ContactID FROM Contact WHERE FirstName = 'Dianne' AND LastName = 'Connor')
  AND EmployeeID = (SELECT EmployeeID FROM Employee WHERE FirstName = 'Jack' AND LastName = 'Lee');


/*
7) Write the SQL SELECT query that displays the names of the employees that
have contacted Toll Brothers (one statement). Run the SQL SELECT query in
MySQL Workbench. Copy the results below as well.
*/
SELECT Employee.FirstName, Employee.LastName
FROM Employee
JOIN ContactEmployee ON Employee.EmployeeID = ContactEmployee.EmployeeID
JOIN Contact ON Contact.ContactID = ContactEmployee.ContactID
JOIN Company ON Contact.CompanyID = Company.CompanyID
WHERE Company.CompanyName = 'Toll Brothers';







 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 