#include<iostream>
using namespace std;

struct Panel_Node {
  Panel_Node(int initial)
  : num_elements(0)
  , next_(nullptr)
  , ptr_(new double[initial]) { }

  int num_elements;
  Panel_Node* next_;
  double* ptr_;
  
};

struct Panel {
public:
  Panel_Node *head_; 
  Panel_Node *tail_; 
  int initial_; 
  int panels_; 
  int panelsize_; 
  double* ptr_;
  int length_;
  int reserved_;

  Panel(int initial) 
    : head_ (NULL)
    , tail_ (NULL)
    , panels_(0)
    , ptr_(new double[initial])
    , length_(0)
    , reserved_(initial) { }

  ~Panel() {
    Panel_Node *current = head_;
    while(current) {
      Panel_Node *temp = current;
      current = current->next_;
      delete temp;
    }
  }

  void
    append(double datum) {
      if ((length_/(panels_+1)) == reserved_) {
        add_panel(datum, reserved_);
      }
      fill_panel(datum);
      length_++;
    }


  void fill_panel(double datum) {
    if (head_ == NULL) { 
      head_ = new Panel_Node(reserved_); 
      head_->ptr_[0] = datum; 
      head_->num_elements++;
      tail_ = head_;
      return;
    }
    if (tail_->num_elements < reserved_) {
      tail_->ptr_[tail_->num_elements] = datum;
      tail_->num_elements++;  
    }
  }
  
  void add_panel(double datum, int reserved_) {
    panels_++; 
    Panel_Node *new_Panel = new Panel_Node(reserved_);  
    new_Panel->num_elements = 0;
    tail_->next_ = new_Panel;
    tail_ = new_Panel;
  }
   

  int get_length()  
  {  
    return length_;  
  }  

  int get_panels()  
  {  
    return panels_+1;  
  }  
   // function for printing the list
   void print()
   {
       Panel_Node *temp = head_;
       while (temp != NULL) {
           for (int i = 0; i < temp->num_elements; i++)
               cout<<(temp->ptr_[i])<<" ";
           cout<<endl;
           temp = temp->next_;
       } 
       cout<<endl<<"Total Number of Panels: "<<get_panels();
       cout<<endl<<"Total Length: "<<get_length();
   }
};

int main() {
   
    int data_size = 13;
    double data[13] = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.1, 2.2, 2.3, 2.4};
    
    int initial = 4;
    Panel p(initial);

    for (int i = 0; i < data_size; i++) {
         p.append(data[i]);
    }
    p.print();
    
    return 0;

};