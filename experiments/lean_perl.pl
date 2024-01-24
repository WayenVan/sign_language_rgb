#! /usr/bin/perl
my $string = "hello_world+and example-string";

$string =~ s/([\+])/hahah/g;
print "$string";
