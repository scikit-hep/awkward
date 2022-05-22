#include<iostream>

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
      fill_panel(datum, reserved_);
      length_++;
    }

  void snapshot() { 
    double* ptr(new double[length_]);  
    Panel_Node *temp = head_;
    int total_length = 0;
    while (temp != NULL) {
      for (int i = 0; i < temp->num_elements; i++) {
        ptr[total_length] = temp->ptr_[i];
        total_length++;
      }
      temp = temp->next_;
    }
    ptr_ = std::move(ptr); 
  }
    
  double
    getitem_at_nowrap(int at) const {
      return ptr_[at];
  }


  void fill_panel(double datum, int reserved) {
    if (head_ == NULL) { 
      head_ = new Panel_Node(reserved); 
      head_->ptr_[0] = datum; 
      head_->num_elements++;
      tail_ = head_;
      return;
    }
    if (tail_->num_elements < reserved) {
      tail_->ptr_[tail_->num_elements] = datum;
      tail_->num_elements++;  
    }
  }
  
  void add_panel(double datum, int reserved) {
    panels_++; 
    Panel_Node *new_Panel = new Panel_Node(reserved);  
    new_Panel->num_elements = 0;
    tail_->next_ = new_Panel;
    tail_ = new_Panel;
  } 

  int length()  
  {  
    return length_;  
  }  

  int panels()  
  {  
    return panels_+1;  
  }  

  Panel_Node *head_; 
  Panel_Node *tail_; 
  int initial_; 
  int panels_; 
  int panelsize_; 
  double* ptr_;
  int length_;
  int reserved_;
};

int main() {
   
    int data_size = 13;
    double data[13] = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.1, 2.2, 2.3, 2.4};
    
    int initial = 10;
    Panel p(initial);

    for (int i = 0; i < data_size; i++) {
         p.append(data[i]);
    }
    p.snapshot();
    for (int at = 0; at < p.length(); at++) {
        std::cout << p.getitem_at_nowrap(at) << ", ";
    }
    std::cout << "\nLength = " << p.length();
    std::cout << "\nPanels = " << p.panels();
    
    return 0;
};