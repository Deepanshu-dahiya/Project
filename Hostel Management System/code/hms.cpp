#include<stdio.h>
#include<stdlib.h>
#include<windows.h>
#define MAX 5
void display();
void badminton_issue(char roll_number[]);
void cock_issue(char roll_number[]);
void tt_issue(char roll_number[]);
void ball_issue(char roll_number[]);
void badminton_return(char roll_number[]);
void cock_return(char roll_number[]);
void tt_return(char roll_number[]);
void ball_return(char roll_number[]);
void issue_item();
void return_item();
char QUEUE_bad[MAX][20];
char QUEUE_ball[MAX][20];
char QUEUE_cock[MAX][20];
char QUEUE_tt[MAX][20];
int front_b=-1,rear_b=-1;
int front_tt=-1,rear_tt=-1;
int front_cock=-1,rear_cock=-1;
int front_ball=-1,rear_ball=-1;
int badminton1=4;
int cock1=4;
int ball1=4;
int tt1=4;
void new_issue(); 
void view();
void delete_issue();
void insert_by_priority(int);
void delete_by_priority(int);
void create();
void check(int);
void display_pqueue();
char queue[5][50]={"Lights and Fans are malfunctioning","TV not working","WiFi not working","Water shortage","Short circuit in hostel"};
int pri_que[MAX];
int front, rear;
void details();
void Common_Room();
void fine();
void issues();
void dots(int x);
void mess();
void category();
void cr_fine(char roll_number[]);
void mess_fine(char roll_number[]);
void hostel_fine(char roll_number[]);
void miscellaneous_fine(char roll_number[]);
void tt(char roll_number[]);
void badminton(char roll_number[]);
void cock(char roll_number[]);
void ball(char roll_number[]);
void glass(char roll_number[]);
void jug(char roll_number[]);
void plate(char roll_number[]);
void spoon(char roll_number[]);

struct node{
	char name[20];
	char roll_no[20];
	char phone_number[20];
	char parents_number[20];
	char home_address[20];
	struct node* next=NULL;
	int hostel_balance;
	struct node* front_entry_next;
	struct node* home_entry_next;
};
struct node* head = NULL;
int main_exit;


void insert(char name[],char roll_no[],char phone_number[],char parents_number[],char home_address[],int hostel_balance){
	struct node* temp=(struct node*)malloc((sizeof(struct node)));
	strcpy(temp->name,name);
	strcpy(temp->roll_no,roll_no);
	strcpy(temp->phone_number,phone_number);
	strcpy(temp->parents_number,parents_number);
	strcpy(temp->home_address,home_address);
	temp->hostel_balance = hostel_balance;
	temp->next=NULL;
	struct node* temp1=head;
	if(head==NULL){
	    head=temp;
	    return;
	}
	while(temp1->next!=NULL){
		temp1=temp1->next;
	}
	temp1->next=temp;
	return;
}

void insert_in_Q_bad(char queue[][20],char ele[])
{
    if(rear_b==-1)
    {
        front_b=rear_b=0;
        strcpy(queue[rear_b],ele);
    }
    else if(rear_b==MAX-1)
    {
        printf("\n\t\t\t\t Sorry, the queue is already full.\n");
        return;
    }
    else
    {
        rear_b++;
        strcpy(queue[rear_b],ele);
    }
    printf("\n\t\t\t\t %s inserted in the queue\n",ele);
}

void display_Q_bad(char queue[][20])
{       
    int i;
    if(rear_b==-1) { printf("\nQUEUE is Empty.\n"); return; }
    for(i=front_b;i<=rear_b;i++)
    { printf(" %s",queue[i]); }

}

void remove_from_Q_bad(char queue[][20])
{
    char ele[20];
    if(front_b==-1)
    {
        printf("\n\t\t\t\t QUEUE is Empty.\n");
        return;
    }
    else if(front_b==rear_b)
    {
        strcpy(ele,queue[front_b]);
        front_b=rear_b=-1;
    }
    else
    {
        strcpy(ele,queue[front_b]);
        front_b++;
    }
    printf("\n\t\t\t\t Item removed: %s.\n",ele);
}

void insert_in_Q_cock(char queue[][20],char ele[])
{
    if(rear_cock==-1)
    {
        front_cock=rear_cock=0;
        strcpy(queue[rear_cock],ele);
    }
    else if(rear_cock==MAX-1)
    {
        printf("\n\t\t\t\t Sorry, the queue is already full.\n");
        return;
    }
    else
    {
        rear_cock++;
        strcpy(queue[rear_cock],ele);
    }
    printf("\t\t\t\t %s inserted in the queue\n",ele);
}

void display_Q_cock(char queue[][20])
{       
    int i;
    if(rear_cock==-1) { printf("\nQUEUE is Empty.\n"); return; }
    for(i=front_cock;i<=rear_cock;i++)
    { printf(" %s",queue[i]); }

}

void remove_from_Q_cock(char queue[][20])
{
    char ele[20];
    if(front_cock==-1)
    {
        printf("\n\n\n\t\t\t\t QUEUE is Empty.\n");
        return;
    }
    else if(front_cock==rear_cock)
    {
        strcpy(ele,queue[front_cock]);
        front_cock=rear_cock=-1;
    }
    else
    {
        strcpy(ele,queue[front_cock]);
        front_cock++;
    }
    printf("\n\t\t\t\t Item removed: %s.\n",ele);
}


void insert_in_Q_tt(char queue[][20],char ele[])
{
    if(rear_tt==-1)
    {
        front_tt=rear_tt=0;
        strcpy(queue[rear_tt],ele);
    }
    else if(rear_tt==MAX-1)
    {
        printf("\t\t\t\t Sorry, the queue is already full.\n");
        return;
    }
    else
    {
        rear_tt++;
        strcpy(queue[rear_tt],ele);
    }
    printf("\t\t\t\t %s inserted in the queue\n",ele);
}

void display_Q_tt(char queue[][20])
{       
    int i;
    if(rear_tt==-1) { printf("\nQUEUE is Empty.\n"); return; }
    for(i=front_tt;i<=rear_tt;i++)
    { printf(" %s",queue[i]); }

}

void remove_from_Q_tt(char queue[][20])
{
    char ele[20];
    if(front_tt==-1)
    {
        printf("\n\n\n\t\t\t\t QUEUE is Empty.\n");
        return;
    }
    else if(front_tt==rear_tt)
    {
        strcpy(ele,queue[front_tt]);
        front_tt=rear_tt=-1;
    }
    else
    {
        strcpy(ele,queue[front_tt]);
        front_tt++;
    }
    printf("\n\t\t\t\t Item removed: %s.\n",ele);
}

void insert_in_Q_ball(char queue[][20],char ele[])
{
    if(rear_ball==-1)
    {
        front_ball=rear_ball=0;
        strcpy(queue[rear_ball],ele);
    }
    else if(rear_ball==MAX-1)
    {
        printf("\t\t\t\t Sorry, the queue is already full.\n");
        return;
    }
    else
    {
        rear_ball++;
        strcpy(queue[rear_ball],ele);
    }
    printf("\t\t\t\t %s inserted in the queue\n",ele);
}

void display_Q_ball(char queue[][20])
{       
    int i;
    if(rear_ball==-1) { printf("\nQUEUE is Empty.\n"); return; }
    for(i=front_ball;i<=rear_ball;i++)
    { printf(" %s",queue[i]); }

}

void remove_from_Q_ball(char queue[][20])
{
    char ele[20];
    if(front_ball==-1)
    {
        printf("\n\n\n\t\t\t\t QUEUE is Empty.\n");
        return;
    }
    else if(front_ball==rear_ball)
    {
        strcpy(ele,queue[front_ball]);
        front_ball=rear_ball=-1;
    }
    else
    {
        strcpy(ele,queue[front_ball]);
        front_ball++;
    }
    printf("\n\t\t\t\t Item removed: %s.\n",ele);
}

void issue_item(){
	mess_try:
		system("cls");
    	system("color 2");
        char roll_number[20];
        printf("Enter the roll number: ");
        scanf("%s",roll_number);
        int flag=0;
        int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       flag+=1;
		}
	    temp = temp->next;
        }
        if(flag==0){
        	char c[1];
	        printf("\ninvalid roll number!\n\n");
	        printf("Enter 1 to retry, 0 to exit or any other key to return to the main menu: ");
            scanf("%s",&c);
            if (c[0]=='1')
                {
					system("cls");
                    goto mess_try;
                }

            else if (c[0]=='0')
                {
                    system("cls");
                    exit(0);}
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
            }
        else{
        	choice_try:
				system("cls");
                system("color 4");
                int choice;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE ONE OF THE FOLLOWING:");
             	printf("\n\n\n\n\n\t\t\t\t 1. Badminton Racket\t\t\t2. Shuttlecock\t\t\t3. TT Bat\t\t4. TT Ball");
             	printf("\n\n\n\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&choice);
             	switch(choice)
                   {
                          case 1:badminton_issue(roll_number);
                          break;
                          case 2:cock_issue(roll_number);
                          break;
                          case 3:tt_issue(roll_number);
                          break;
                          case 4:ball_issue(roll_number);
                          break;
                          default:{
                          	printf("\n\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto choice_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto choice_try;}
									}
                  } 
	            
                                }
}
void return_item(){
	mess_try:
		system("cls");
    	system("color 2");
        char roll_number[20];
        printf("Enter the roll number: ");
        scanf("%s",roll_number);
        int flag=0;
        int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       flag+=1;
		}
	    temp = temp->next;
        }
        if(flag==0){
        	char c[1];
	        printf("\ninvalid roll number!\n\n");
	        printf("Enter 1 to retry, 0 to exit or any other key to return to the main menu: ");
            scanf("%s",&c);
            if (c[0]=='1')
                {
					system("cls");
                    goto mess_try;
                }

            else if (c[0]=='0')
                {
                    system("cls");
                    exit(0);}
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
            }
        else{
        	choice_try:
				system("cls");
                system("color 4");
                int choice;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE ONE OF THE FOLLOWING:");
             	printf("\n\n\n\n\n\t\t\t\t 1. Badminton Racket\t\t\t2. Shuttlecock\t\t\t3. TT Bat\t\t4. TT Ball");
             	printf("\n\n\n\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&choice);
             	switch(choice)
                   {
                          case 1:badminton_return(roll_number);
                          break;
                          case 2:cock_return(roll_number);
                          break;
                          case 3:tt_return(roll_number);
                          break;
                          case 4:ball_return(roll_number);
                          break;
                          default:{
                          	printf("\n\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto choice_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto choice_try;}
									}
                  } 
	            
                                }
}

void display(){
	system("cls");
    system("color 2");
	printf("Queue for badminton:");
	display_Q_bad(QUEUE_bad);
	printf("\n");
	printf("Queue for shuttlecock:");
	display_Q_cock(QUEUE_cock);
	printf("\n");
	printf("Queue for TT bat:");
	display_Q_tt(QUEUE_tt);
	printf("\n");
	printf("Queue for TT ball:");
	display_Q_ball(QUEUE_ball);
	printf("\n");
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
};

void badminton_return(char roll_number[]){
	if(badminton1!=4){
	 badminton1++;
	 remove_from_Q_bad(QUEUE_bad);
	 printf("\n\n\n\t\t\t\t Item returned by %s",roll_number);
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	printf("\n\n\n\t\t\t\t This item was not issued.");
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}}

void cock_return(char roll_number[]){
	if(cock1!=4){
	 cock1++;
	 remove_from_Q_cock(QUEUE_cock);
	 printf("\n\n\n\t\t\t\t Item returned by %s",roll_number);
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	printf("\n\n\n\t\t\t\t This item was not issued.");
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}}

void ball_return(char roll_number[]){
	if(ball1!=4){
	 ball1++;
	 remove_from_Q_ball(QUEUE_ball);
	 printf("\n\n\n\t\t\t\t Item returned by %s",roll_number);
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	printf("\n\n\n\t\t\t\t This item was not issued.");
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}}

void tt_return(char roll_number[]){
	if(tt1!=4){
	 tt1++;
	 remove_from_Q_tt(QUEUE_tt);
	 printf("\n\n\n\t\t\t\t Item returned by %s",roll_number);
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	printf("\n\n\n\t\t\t\t This item was not issued.");
	 printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}}

void badminton_issue(char roll_number[]){
	if(badminton1!=0){
	 badminton1--;
	 printf("\n\n\n\t\t\t\t Item issued to %s",roll_number);
	 printf("\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	system("cls");
    system("color 2");
    printf("The selected item is not available\n\n");
    printf("Would you like to become part of the queue\n\n");
    printf("Enter 1 for yes or any other key to return to the main menu: ");
            char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					printf("\n\n");
                    insert_in_Q_bad(QUEUE_bad,roll_number);
                    printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
                }
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
}
}

void cock_issue(char roll_number[]){
	if(cock1!=0){
	 cock1--;
	 printf("\n\n\n\t\t\t\t Item issued to %s",roll_number);
	 printf("\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	system("cls");
    system("color 2");
    printf("The selected item is not available\n\n");
    printf("Would you like to become part of the queue\n\n");
    printf("Enter 1 for yes or any other key to return to the main menu: ");
            char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					printf("\n\n");
                    insert_in_Q_cock(QUEUE_cock,roll_number);
                    printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
                }
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
}
}

void ball_issue(char roll_number[]){
	if(ball1!=0){
	 ball1--;
	 printf("\n\n\n\t\t\t\t Item issued to %s",roll_number);
	 printf("\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	system("cls");
    system("color 2");
    printf("The selected item is not available\n\n");
    printf("Would you like to become part of the queue\n\n");
    printf("Enter 1 for yes or any other key to return to the main menu: ");
            char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					printf("\n\n");
                    insert_in_Q_ball(QUEUE_ball,roll_number);
                    printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
                }
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
}
}

void tt_issue(char roll_number[]){
	if(tt1!=0){
	 tt1--;
	 printf("\n\n\n\t\t\t\t Item issued to %s",roll_number);
	 printf("\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
}
else{
	system("cls");
    system("color 2");
    printf("The selected item is not available\n\n");
    printf("Would you like to become part of the queue\n\n");
    printf("Enter 1 for yes or any other key to return to the main menu: ");
            char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					printf("\n\n");
                    insert_in_Q_tt(QUEUE_tt,roll_number);
                    printf("\n\n\t\t\t\t Enter 1 to go to main menu or any other key to exit: ");
                             char c[1];
			scanf("%s",&c);
            if (c[0]=='1')
                {
					category();
                }
            else
                {
                    system("cls");
                    exit(0);
					}
                }
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
}
}


void new_issue(){
	int category1;
	category_try:
                system("cls");
                system("color 2");
             	printf("\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE AMONG THE FOLLOWING ISSUES:");
             	printf("\n\n\n\n\n\t\t\t\t\t\t\t 1. Lights and Fans\t              \t\t\t2. TV");
                printf("\n\n\n\t\t\t\t\t\t\t 3. WiFi                                 \t\t4. Water");
            	printf("\n\n\n\t\t\t\t\t\t\t 5. Short circuit in hostel ");
             	printf("\n\n\n\t\t\t\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&category1);
             	switch(category1)
                   {
                          case 1:insert_by_priority(1);
                          break;
                          case 2:insert_by_priority(2);
                          break;
                          case 3:insert_by_priority(3);
                          break;
                          case 4:insert_by_priority(4);
                          break;
                          case 5:insert_by_priority(5);
                          break;
                          default:{
                          	printf("\n\t\t\t\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto category_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto category_try;}
									}
                  } 
                  char k[1];
                  printf("\n\n\t\t\t\t\t\t\t press 1 to go back, 0 to go to main menu or any other key to exit: ");
                  scanf("%s",&k);
                            if (k[0]=='1')
                                {
                                 system("cls");
                                 issues();
                                }

                            else if (k[0]=='0')
                                   {
                                      system("cls");
									  category();
									  }
                            else
                                   {
								   exit(0);}
                                }


void view(){
	system("cls");
    system("color 2");
	display_pqueue();
	char k[1];
                  printf("\n\n\n\n\n\n\n\t\t\t\t\t\t press 1 to go back, 0 to go to main menu or any other key to exit: ");
                  scanf("%s",&k);
                            if (k[0]=='1')
                                {
                                 issues();
                                }

                            else if (k[0]=='0')
                                   {
                                      system("cls");
									  category();
									  }
                            else
                                   {
								   exit(0);}
}

void delete_issue(){
	int category1;
	category_try:
                system("cls");
                system("color 3");
             	printf("\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE AMONG THE FOLLOWING ISSUES:");
             	printf("\n\n\n\n\n\t\t\t\t\t\t\t 1. Lights and Fans\t              \t\t\t2. TV");
                printf("\n\n\n\t\t\t\t\t\t\t 3. WiFi                                 \t\t4. Water");
            	printf("\n\n\n\t\t\t\t\t\t\t 5. Short circuit in hostel ");
             	printf("\n\n\n\t\t\t\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&category1);
             	switch(category1)
                   {
                          case 1:delete_by_priority(1);
                          break;
                          case 2:delete_by_priority(2);
                          break;
                          case 3:delete_by_priority(3);
                          break;
                          case 4:delete_by_priority(4);
                          break;
                          case 5:delete_by_priority(5);
                          break;
                          default:{
                          	printf("\n\t\t\t\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto category_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto category_try;}
									}
                  } 
                  char k[1];
                  printf("\n\n\t\t\t\t\t\t\t press 1 to go back, 0 to go to main menu or any other key to exit: ");
                  scanf("%s",&k);
                            if (k[0]=='1')
                                {
                                 issues();
                                }

                            else if (k[0]=='0')
                                   {
                                      system("cls");
									  category();
									  }
                            else
                                   {
								   exit(0);}
                                }

void create()
{
    front = rear = -1;
}
 

void insert_by_priority(int data)
{
	printf("\n\n\n\t\t\t\t\t\t\t Issue added to the queue.");
    if (rear >= MAX - 1)
    {
        printf("\n\t\t\t\t\t\t\t Queue overflow no more elements can be inserted");
        return;
    }
    if ((front == -1) && (rear == -1))
    {
        front++;
        rear++;
        pri_que[rear] = data;
        return;
    }    
    else
        check(data);
    rear++;
}
 

void check(int data)
{
    int i,j;
 
    for (i = 0; i <= rear; i++)
    {
        if (data >= pri_que[i])
        {
            for (j = rear + 1; j > i; j--)
            {
                pri_que[j] = pri_que[j - 1];
            }
            pri_que[i] = data;
            return;
        }
    }
    pri_que[i] = data;
}
 

void delete_by_priority(int data)
{
    int i;
    printf("\n\n\n\t\t\t\t\t\t\t Issue deleted.");
    if ((front==-1) && (rear==-1))
    {
        printf("\n\t\t\t\t\t\t\t Queue is empty no elements to delete");
        return;
    }
 
    for (i = 0; i <= rear; i++)
    {
        if (data == pri_que[i])
        {
            for (; i < rear; i++)
            {
                pri_que[i] = pri_que[i + 1];
            }
 
        pri_que[i] = -99;
        rear--;
 
        if (rear == -1) 
            front = -1;
        return;
        }
    }
    printf("\n\t\t\t\t\t\t\t %d not found in queue to delete", data);
}
 

void display_pqueue()
{
    if ((front == -1) && (rear == -1))
    {
        printf("\n\t\t\t\t\t\t\t Queue is empty");
        return;
    }
 
    for (; front <= rear; front++)
    {
        printf("%s\n", queue[pri_que[front]-1]);
    }
 
    front = 0;
}

void badminton(char roll_number[]){
	    int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=200;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 200 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
									
}

void plate(char roll_number[]){
	    int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=35;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 35 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
									
}

void spoon(char roll_number[]){
	    int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=7;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 7 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
									
}

void glass(char roll_number[]){
	    int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=15;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 15 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
									
}

void jug(char roll_number[]){
	    int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=20;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 20 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
									
}

void mess_fine(char roll_number[]){
	choice_try:
				system("cls");
                system("color 4");
                int choice;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE THE CATEGORY:");
             	printf("\n\n\n\n\n\t\t\t\t 1. Plates\t\t\t2. Spoons\t\t\t3. Glass\t\t4. Jug");
             	printf("\n\n\n\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&choice);
             	switch(choice)
                   {
                          case 1:plate(roll_number);
                          break;
                          case 2:spoon(roll_number);
                          break;
                          case 3:glass(roll_number);
                          break;
                          case 4:jug(roll_number);
                          break;
                          default:{
                          	printf("\n\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto choice_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto choice_try;}
									}
                  }
	
}

void miscellaneous_fine(char roll_number[]){
	choice_try:
				system("cls");
                system("color 4");
                int amount;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tEnter the amount: ");
             	scanf("%d",&amount);
                int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=amount;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t\t\t\t   Rs %d has been deducted from the total balance.",amount);
        printf("\n\t\t\t\t\t\t\t   Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t\t\t\t   Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t\t   Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
                          	
}
void tt(char roll_number[]){
	int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=100;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 100 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
}
void cock(char roll_number[]){
	int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=50;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 50 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
}
void ball(char roll_number[]){
	int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=30;
	       break;
		}
	    temp = temp->next;
        }
        printf("\n\n\n\t\t\t\t Rs 30 has been deducted from the total balance.");
        printf("\n\t\t\t\t Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n\t\t\t\t Enter 1 to go to the main menu or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    category();}
}
void cr_fine(char roll_number[]){
	choice_try:
				system("cls");
                system("color 4");
                int choice;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE THE CATEGORY:");
             	printf("\n\n\n\n\n\t\t\t\t 1. Badminton Racket\t\t\t2. Shuttlecock\t\t\t3. TT Bat\t\t4. TT Ball");
             	printf("\n\n\n\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&choice);
             	switch(choice)
                   {
                          case 1:badminton(roll_number);
                          break;
                          case 2:cock(roll_number);
                          break;
                          case 3:tt(roll_number);
                          break;
                          case 4:ball(roll_number);
                          break;
                          default:{
                          	printf("\n\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto choice_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto choice_try;}
									}
                  } 
}

void hostel_fine(char roll_number[]){
	choice_try:
				system("cls");
                system("color 4");
                int choice;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE THE CATEGORY:");
             	printf("\n\n\n\n\n\t\t\t\t 1. Tubelight\t\t\t2. Window\t\t\t3. Geyser\t\t4. Water Cooler");
             	printf("\n\n\n\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&choice);
             	switch(choice)
                   {
                          case 1:jug(roll_number);
                          break;
                          case 2:tt(roll_number);
                          break;
                          case 3:badminton(roll_number);
                          break;
                          case 4:badminton(roll_number);
                          break;
                          default:{
                          	printf("\n\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto choice_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto choice_try;}
									}
                  } 
}

void details(){
	details_try:
	    int i;
		system("cls");
    	system("color 2");
	    printf("The details of hostellers are:\n\n");
	    printf("Name        Roll Number         Phone Number            Parents's no.           Address         Balance\n\n");
	    struct node* temp = head;
	    for(i=0;i<=12;i++){
	    	printf("%s%s\t\t%s     \t%s\t\t%s\t%d\n",temp->name,temp->roll_no,temp->phone_number,temp->parents_number,temp->home_address,temp->hostel_balance);
	    	temp = temp->next;
		}
		printf("\n\n");
		printf("Enter 1 to go back or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {
								 system("cls");
                                 category();
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\nInvalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto details_try;}
									
		printf("\n\n\n\n\n\n\n\n\n\n");
	    
}
void Common_Room(){
			system("cls");
            	system("color 2");
	        	
option_try:	
	int option;
	char k[1];
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\tPlease choose one of the following: \n\n\n");
	printf("\n\n\n\n\t\t\t\t\t      1. Issue an item  \t2. Return an item  \t3. Display Queue");
	            printf("\n\n\n\n\t\t\t\t\t      Select one from the above: ");
             	scanf("%d",&option);
             	switch(option)
                   {
                          case 1:issue_item();
                          break;
                          case 2:return_item();
                          break;
                          case 3:display();
                          break;
                          default:{
                          	printf("\n\n\n\n\t\t\t\t\t      invalid input!\a\a\a\n");
                          	printf("\n\n\t\t\t\t\t      Enter 1 to try again or 0 to go to main menu or any other key to exit: ");
                            scanf("%s",&k);
                            if (k[0]=='1')
                                {
								 system("cls");
                                 goto option_try;
                                }

                            else if (k[0]=='0')
                                   {
                                      system("cls");
                                      category();}
                            else
                                   {
                                    system("cls");
                                    exit(0);}
									}
                  }  
}
void mess(){
	mess_try:
		system("cls");
    	system("color 2");
        char roll_number[20];
        printf("Enter the roll number for attendance: ");
        scanf("%s",roll_number);
        int flag=0;
        int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       flag+=1;
		}
	    temp = temp->next;
        }
        if(flag==0){
        	char c[1];
	        printf("\ninvalid roll number!\n\n");
	        printf("Enter 1 to retry, 0 to exit or any other key to return to the main menu: ");
            scanf("%s",&c);
            if (c[0]=='1')
                {
					system("cls");
                    goto mess_try;
                }

            else if (c[0]=='0')
                {
                    system("cls");
                    exit(0);}
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
            }
        else{
	           struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       temp->hostel_balance-=50;
	       break;
		}
	    temp = temp->next;
        }
        char k[1];
        printf("\n\n\n Rs 50 has been deducted from the total balance.");
        printf("\n Current Balance = Rs %d",temp->hostel_balance);
        printf("\n\n Enter 1 to enter another roll number, 0 to go to the main menu or any other key to exit: ");
                            scanf("%s",&k);
                            if (k[0]=='1')
                                {
                                 system("cls");
                                 goto mess_try;
                                }

                            else if (k[0]=='0')
                                   {
                                      system("cls");
                                      category();}
                            else
                                   {
								   exit(0);}
                                }
}
void fine(){
	fine_try:
		system("cls");
    	system("color 2");
        char roll_number[20];
        printf("Enter the roll number of the defaulter: ");
        scanf("%s",roll_number);
        int flag=0;
        int i;
        struct node* temp = head;
        for(i=0;i<=12;i++){
	    if(strcmp(temp->roll_no,roll_number)==0){
	       flag+=1;
		}
	    temp = temp->next;
        }
        if(flag==0){
        	char c[1];
	        printf("\ninvalid roll number!\n\n");
	        printf("Enter 1 to retry, 0 to exit or any other key to return to the main menu: ");
            scanf("%s",&c);
            if (c[0]=='1')
                {
					system("cls");
                    goto fine_try;
                }

            else if (c[0]=='0')
                {
                    system("cls");
                    exit(0);}
            else
                {
                    dots(1000000000);
                    system("cls");
                    category();
					}
            }
        else{
	           choice_try:
				system("cls");
                system("color 2");
                int choice;
                printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE THE CATEGORY:");
             	printf("\n\n\n\n\n\t\t\t\t 1. Common room\t\t\t2. Mess\t\t\t3. Hostel\t\t4. Miscellaneous");
             	printf("\n\n\n\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&choice);
             	switch(choice)
                   {
                          case 1:cr_fine(roll_number);
                          break;
                          case 2:mess_fine(roll_number);
                          break;
                          case 3:hostel_fine(roll_number);
                          break;
                          case 4:miscellaneous_fine(roll_number);
                          break;
                          default:{
                          	printf("\n\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto choice_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto choice_try;}
									}
                  } 
            }
}
void issues(){
			system("cls");
        	system("color 2");
	        	
option_try:	
	int option;
	char k[1];
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\tPlease choose one of the following: \n\n\n");
	printf("\n\n\n\n\t\t\t\t\t      1. Enter a new issue.  \t2. View issue list  \t3. Delete an issue");
	            printf("\n\n\n\n\t\t\t\t\t      Select one from the above: ");
             	scanf("%d",&option);
             	switch(option)
                   {
                          case 1:new_issue();
                          break;
                          case 2:view();
                          break;
                          case 3:delete_issue();
                          default:{
                          	printf("\n\n\n\n\t\t\t\t\t      invalid input!\a\a\a\n");
                          	printf("\n\n\t\t\t\t\t      Enter 1 to try again or 0 to go to main menu or any other key to exit: ");
                            scanf("%s",&k);
                            if (k[0]=='1')
                                {
								 system("cls");
                                 goto option_try;
                                }

                            else if (k[0]=='0')
                                   {
                                      system("cls");
                                      category();}
                            else
                                   {
                                    system("cls");
                                    exit(0);}
									}
                  } 
}

void dots(int j)
{   int i,k;
    for(i=0;i<j;i++)
         k=i;
}

void category(){
	int category1;
	category_try:
                system("cls");
                system("color 3");
             	printf("\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tCHOOSE THE CATEGORY:");
             	printf("\n\n\n\n\n\t\t\t\t\t\t 1. Details of a student            \t\t2. Mess entry system");
                printf("\n\n\n\t\t\t\t\t\t 3. Common Room Management              \t4. Issues");
            	printf("\n\n\n\t\t\t\t\t\t 5. Fine system");
             	printf("\n\n\n\t\t\t\t\t\t ");
             	printf("Please choose your category: ");
             	scanf("%d",&category1);
             	switch(category1)
                   {
                          case 1:details();
                          break;
                          case 2:mess();
                          break;
                          case 3:Common_Room();
                          break;
                          case 4:issues();
                          break;
                          case 5:fine();
                          break;
                          default:{
                          	printf("\n\t\t\t\t\t\t invalid input!\a\a\a\n");
                          	printf("\n\t\t\t\t\t\t Enter 1 to try again or 0 to exit: ");
                            scanf("%d",&main_exit);
                            if (main_exit==1)
                                {

                                 system("cls");
                                 goto category_try;
                                }

                            else if (main_exit==0)
                                   {
                                      system("cls");
                                      exit(0);}
                            else
                                   {
								    printf("\n\t\t\t\t\t\t Invalid!");
                                    dots(1000000000);
                                    system("cls");
                                    goto category_try;}
									}
                  } 
}

int main(){
	create();
	int o;
	char name[13][20] = {"Pranjal     ","Deepanshu   ","Anuvrat     ","Jay         ","Arun        ","Kashish     ","Tushar      ","Ojaswi      ","Harshit     ","Tushant     ","Prashant    ","Vishal      ","Piyush      "};
	char roll_no[13][20] = {"18306","18310","18109","18301","18335","18223","18333","18236","18235","18240","18139","18129","18231"};
	char phone_no[13][20] = {"1830600000  ","1831000000  ","1810900000  ","1830100000  ","1833500000  ","1822300000  ","1833300000  ","1823600000  ","1823500000  ","1824000000  ","1813900000  ","1812900000  ","1823100000  "};
	char parents_no[13][20] = {"1830611111  ","1831011111  ","1810911111  ","1830111111  ","1833511111  ","1822311111  ","1833311111  ","1823611111  ","1823511111  ","1824011111  ","1813911111  ","1812911111  ","1823111111  "};
	char address[13][20] = {"Nainital    ","Rohtak      ","Meerut      ","Gwalior     ","Gurugram    ","Chandigarh  ","Delhi       ","Saharanpur  ","Saharanpur  ","Meerut      ","Una         ","Delhi       ","Chandigarh  "};
	int balance[13] = {23000,23000,23000,23000,23000,23000,23000,23000,23000,23000,23000,23000,23000};
	for(o=0;o<=12;o++){
	    	insert(name[o],roll_no[o],phone_no[o],parents_no[o],address[o],balance[o]);
	}
	char c;
	
	system("color 6");
	printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\tWELCOME TO HOSTEL MANAGEMENT SYSTEM\n\n\t\t\t\t\t\t\t\tpress enter to continue...\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
	scanf("%c",&c);
	if(c){
	char pass[10],password[10]="123";
    int i=0;
    system("cls");
    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\tEnter the password to login: ");
    scanf("%s",pass);
    if (strcmp(pass,password)==0)
        {
        system("cls");	
		printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tPassword Match!\n\t\t\t\t\t\t\t\t\tLOADING");
        for(i=0;i<=6;i++)
        {
            dots(100000000);
            printf(".");
        }
            category();
        }
    else
        {   printf("\n\t\t\t\t\t\t\t\tWrong password!!\a\a\a\n");
            login_try:
            printf("\n\t\t\t\t\t\t\t\tEnter 1 to try again or 0 to exit: ");
            scanf("%d",&main_exit);
            if (main_exit==1)
                    {

                        system("cls");
                        main();
                    }

            else if (main_exit==0)
                    {
                    system("cls");
                    exit(0);}
            else
                    {printf("\n\t\t\t\t\t\t\t\tInvalid!");
                    dots(1000000000);
                    system("cls");
                    goto login_try;}

        }
	}
	return 0;
}
